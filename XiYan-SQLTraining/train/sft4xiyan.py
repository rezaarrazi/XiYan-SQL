import os
import sys
sys.path.append('../')
import json
import numpy as np
import transformers
from transformers import (
    Trainer,
    GenerationConfig,
    set_seed
)
from trainer.trainer import DeepCustomTrainer
from trainer.argument import ModelArguments, TrainingArguments, DataArguments, LoraArguments
from trainer.train_util import load_tokenizer_and_model, DataCollatorForGeneration
from datasets import load_dataset, concatenate_datasets
try:
    from datasets import set_caching_enabled
    set_caching_enabled(True)
except ImportError:
    # datasets >= 3.0 uses enable_caching/disable_caching
    from datasets import enable_caching
    enable_caching()
from utils.common_utils import read_json, write_json

# Use swanlab to visualize training records
import swanlab
from swanlab.integration.transformers import SwanLabCallback

IGNORE_TOKEN_ID = -100
# Global variables
call_counter = 0


def clean_sql_response(content):
    """Extract only SQL from response, removing any explanations."""
    if not content:
        return content
    
    content = content.strip()
    
    # If wrapped in markdown code blocks, extract SQL
    if '```sql' in content:
        parts = content.split('```sql')
        if len(parts) > 1:
            sql_part = parts[1].split('```')[0].strip()
            return sql_part
    
    # If wrapped in plain code blocks
    if '```' in content:
        parts = content.split('```')
        if len(parts) > 1:
            sql_part = parts[1].strip()
            # Check if it looks like SQL
            if any(sql_part.upper().startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']):
                return sql_part
    
    # If starts with SQL keywords, take until explanation starts
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
    for keyword in sql_keywords:
        if content.upper().startswith(keyword):
            # Find the SQL part (until explanation starts)
            lines = content.split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop if we hit explanation patterns
                explanation_patterns = ['this query', 'here\'s', 'to find', 'you can use', 
                                      'the query', 'this will', 'selects', 'groups', 
                                      'This query', 'Here\'s', 'To find']
                if any(pattern in line for pattern in explanation_patterns):
                    # Only break if we already have some SQL
                    if len(sql_lines) > 0:
                        break
                sql_lines.append(line)
            return ' '.join(sql_lines)
    
    return content


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    # set seed
    set_seed(training_args.seed)
    tokenizer, model = load_tokenizer_and_model(model_args, training_args, lora_args)

    def preprocess_data(example):
        conversations = example['conversations']
        
        # Clean the target SQL to remove any explanations
        target = clean_sql_response(conversations[1]['content'])
        
        # Get prompt part (user message only) - this is what we want to mask
        prompt_part = tokenizer.apply_chat_template(
            conversations[:1],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Get full text (user + assistant) for training
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize both to get accurate positions
        prompt_encodings = tokenizer(prompt_part, truncation=True, return_offsets_mapping=False)
        full_encodings = tokenizer(text, truncation=True, return_offsets_mapping=False)
        
        # Find where assistant response starts (after prompt)
        target_idx = len(prompt_encodings['input_ids'])
        
        labels = full_encodings['input_ids'].copy()
        
        # Mask everything before the assistant response
        if target_idx:
            labels[:target_idx] = [IGNORE_TOKEN_ID] * target_idx
        
        # Ensure labels match input_ids length (handle truncation edge cases)
        if len(labels) != len(full_encodings['input_ids']):
            min_len = min(len(labels), len(full_encodings['input_ids']))
            labels = labels[:min_len]
            full_encodings['input_ids'] = full_encodings['input_ids'][:min_len]
        
        assert len(labels) == len(full_encodings['input_ids'])
        # Route different dialects under momq, ignoring in normal mode.
        if training_args.enable_dialect_router:
            sql_type = example.get("sql_type", "sqlite").lower()
            if sql_type == 'postgresql':
                labels[0] = min(0, training_args.dialect_num - 1)
            elif sql_type == 'mysql':
                labels[0] = min(1, training_args.dialect_num - 1)
            elif sql_type == 'sqlite':
                labels[0] = min(2, training_args.dialect_num - 1)
            elif sql_type == 'cypher':
                labels[0] = min(3, training_args.dialect_num - 1)
            elif sql_type == 'ngql':
                labels[0] = min(4, training_args.dialect_num - 1)
        full_encodings['labels'] = labels
        return full_encodings

    def preprocess_eval_data(example):
        conversations = example['conversations'][:1]
        text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            # chat_template=TEMPLATE,
            add_generation_prompt=True
        )
        encodings = tokenizer(text, truncation=True)
        db_name = example.get('db_name', '')
        sql_type = example.get("sql_type", "postgresql")
        target = example['conversations'][1]['content']
        labels = tokenizer(db_name + '[sep]' + sql_type + '[sep]' + target, truncation=True)['input_ids']
        encodings['labels'] = labels
        return encodings

    # load data
    train_data = load_dataset('json', data_files=data_args.data_path)['train']
    train_data = train_data.map(preprocess_data, remove_columns=train_data.column_names, num_proc=16, load_from_cache_file=True)
    
    # Shuffle training data if requested
    if data_args.do_shuffle:
        print(f"🔀 Shuffling training data with seed={training_args.seed}")
        train_data = train_data.shuffle(seed=training_args.seed)
        print(f"✅ Training data shuffled ({len(train_data)} samples)")
    
    eval_data = None
    if data_args.eval_data_path:
        eval_data_raw = load_dataset('json', data_files=data_args.eval_data_path)['train']
        eval_data = eval_data_raw.map(preprocess_eval_data, remove_columns=eval_data_raw.column_names, num_proc=16, load_from_cache_file=True)
    data_collator = DataCollatorForGeneration(tokenizer)

    def save_and_evaluate(pred_sqls, label_sqls):
        global call_counter
        call_counter += 1
        eval_data_result = []
        for idx in range(len(label_sqls)):
            pred_sql = pred_sqls[idx]
            new_item = {
                "idx": idx,
                "pred_sql": pred_sql,
                "sql": label_sqls[idx]
            }
            eval_data_result.append(new_item)
        # The calculation of metrics is omitted here.
        metrics = {}

        # save results
        save_path = os.path.join(
            training_args.output_dir,
            training_args.expr_id,
            f"{training_args.expr_id}_{int(call_counter * training_args.eval_steps):05d}.json"
        )
        write_json(save_path, eval_data_result)
        return metrics

    def compute_metrics(eval_pred):
        predictions, labels, inputs = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        gt_sqls = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_sqls = [
            pred[len(prompt):] for prompt, pred in zip(decoded_inputs, decoded_preds)
        ]
        metrics = save_and_evaluate(pred_sqls, gt_sqls)
        return metrics

    # Only used for evaluation during training
    training_args.generation_config = GenerationConfig(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        temperature=None,
        num_beams=1,
        top_p=None,
        do_sample=False,
        use_cache=True
    )
    # Add extra loss for momq-moe
    extra_losses = []
    if training_args.enable_dialect_router:
        extra_losses.append('dialect_loss')
    if training_args.output_router_logits:
        extra_losses.append('aux_loss')

    # SwanLabCallback
    swanlab_callback = SwanLabCallback(
        project="SQLTrainer",
        experiment_name=training_args.expr_id
    )
    # Start via Deeply customized trainer
    trainer = DeepCustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        extra_losses=extra_losses,
        callbacks=[swanlab_callback],
    )

    # start via hf native trainer
    # trainer = Trainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=train_data,
    #     eval_dataset=eval_data,
    #     data_collator=data_collator,
    #     callbacks = [swanlab_callback],
    # )

    trainer.train(resume_from_checkpoint=training_args.resume)
    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    trainer.save_state()
    trainer.save_model()

if __name__ == "__main__":
    train()

