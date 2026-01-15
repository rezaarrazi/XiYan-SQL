import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os

# Use the same template format as training
NL2SQLITE_TEMPLATE_EN = """You are a SQLite expert. You need to read and understand the following【Database Schema】description and the possible provided【Evidence】, and use valid SQLite knowledge to generate SQL for answering the【Question】.
【Question】
{question}

【Database Schema】
{db_schema}

【Evidence】
{evidence}

【Question】
{question}

```sql"""


def extract_sql_only(text):
    """Extract only SQL from model output, removing explanations."""
    if not text:
        return text
    
    text = text.strip()
    
    # Pattern 1: SQL in markdown code blocks
    if '```sql' in text:
        parts = text.split('```sql')
        if len(parts) > 1:
            sql = parts[1].split('```')[0].strip()
            return sql
    
    # Pattern 2: SQL in plain code blocks
    if '```' in text:
        parts = text.split('```')
        if len(parts) > 1:
            sql = parts[1].strip()
            # Check if it looks like SQL
            if any(sql.upper().startswith(kw) for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']):
                return sql
    
    # Pattern 3: SQL starts with SELECT/INSERT/etc
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'WITH']
    for keyword in sql_keywords:
        if text.upper().startswith(keyword):
            # Take until we hit explanation or end
            lines = text.split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop at explanation patterns
                explanation_indicators = [
                    'this query', 'here\'s', 'to find', 'you can use',
                    'the query', 'this will', 'selects', 'groups',
                    'This query', 'Here\'s', 'To find', 'You can use',
                    'The query', 'This will'
                ]
                # Check if line contains explanation (but not if it's actual SQL)
                is_explanation = any(indicator in line.lower() for indicator in explanation_indicators)
                # Also check for markdown code block endings that suggest explanation follows
                if is_explanation or ('```' in line and 'SELECT' not in line.upper()):
                    # Only break if we already have SQL
                    if len(sql_lines) > 0:
                        break
                sql_lines.append(line)
            return ' '.join(sql_lines)
    
    # Pattern 4: Look for SQL after explanation text
    # Find first occurrence of SQL keywords
    for keyword in sql_keywords:
        idx = text.upper().find(keyword)
        if idx != -1:
            # Extract from that position
            sql_text = text[idx:].strip()
            lines = sql_text.split('\n')
            sql_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Stop at new explanation
                explanation_indicators = [
                    'this query', 'here\'s', 'to find', 'you can use',
                    'the query', 'this will'
                ]
                if any(indicator in line.lower() for indicator in explanation_indicators):
                    if len(sql_lines) > 0:
                        break
                sql_lines.append(line)
            return ' '.join(sql_lines)
    
    return text


def run_quick_test(model_path, adapter_path=None):
    print(f"🎯 Testing model: {model_path}")
    if adapter_path:
        print(f"   Adapter: {adapter_path}")
    
    print("\n📥 Loading model and tokenizer...")
    print("   (This may take 1-2 minutes)\n")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        if adapter_path:
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            # Load adapter on top
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.bfloat16
            )
            model.eval()
        else:
            # Load model directly
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            model.eval()
        
        print("✅ Model loaded successfully!\n")
        
        # Test cases (using m-schema format like training data)
        # All from movie_3 database
        movie_3_schema = """【DB_ID】 movie_3
【Schema】
# Table: film
[
(film_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(title:TEXT, Examples: [ACADEMY DINOSAUR, ACE GOLDFINGER, ADAPTATION HOLES]),
(description:TEXT),
(release_year:TEXT, Examples: [2006]),
(language_id:INTEGER, Examples: [1]),
(original_language_id:INTEGER),
(rental_duration:INTEGER, Examples: [6, 3, 7]),
(rental_rate:REAL, Examples: [0.99, 4.99, 2.99]),
(length:INTEGER, Examples: [86, 48, 50]),
(replacement_cost:REAL, Examples: [20.99, 12.99, 18.99]),
(rating:TEXT, Examples: [PG, G, NC-17]),
(special_features:TEXT, Examples: [Trailers,Deleted Scenes, Commentaries,Behind the Scenes]),
(last_update:DATETIME, Examples: [2006-02-15 05:03:42.0])
]
# Table: rental
[
(rental_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(rental_date:DATETIME, Examples: [2005-05-24 22:53:30.0]),
(inventory_id:INTEGER, Examples: [367, 1525, 1711]),
(customer_id:INTEGER, Examples: [130, 459, 408]),
(return_date:DATETIME, Examples: [2005-05-26 22:04:30.0]),
(staff_id:INTEGER, Examples: [1, 2]),
(last_update:DATETIME, Examples: [2006-02-15 21:30:53.0])
]
# Table: store
[
(store_id:INTEGER, Primary Key, Examples: [1, 2]),
(manager_staff_id:INTEGER, Examples: [1, 2]),
(address_id:INTEGER, Examples: [1, 2]),
(last_update:DATETIME, Examples: [2006-02-15 04:57:12.0])
]
# Table: inventory
[
(inventory_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(film_id:INTEGER, Examples: [1, 2, 3]),
(store_id:INTEGER, Examples: [1, 2]),
(last_update:DATETIME, Examples: [2006-02-15 05:09:17.0])
]
# Table: address
[
(address_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(address:TEXT, Examples: [47 MySakila Drive, 28 MySQL Boulevard, 23 Workhaven Lane]),
(address2:TEXT),
(district:TEXT, Examples: [Alberta, QLD, Nagasaki]),
(city_id:INTEGER, Examples: [300, 576, 463]),
(postal_code:TEXT, Examples: [35200, 17886, 83579]),
(phone:TEXT, Examples: [14033335568, 6172235589, 28303384290]),
(last_update:DATETIME, Examples: [2006-02-15 04:45:30.0])
]
# Table: country
[
(country_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(country:TEXT, Examples: [Afghanistan, Algeria, American Samoa]),
(last_update:DATETIME, Examples: [2006-02-15 04:44:00.0])
]
# Table: city
[
(city_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(city:TEXT, Examples: [A Corua (La Corua), Abha, Abu Dhabi]),
(country_id:INTEGER, Examples: [87, 82, 101]),
(last_update:DATETIME, Examples: [2006-02-15 04:45:25.0])
]
# Table: film_actor
[
(actor_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(film_id:INTEGER, Primary Key, Examples: [1, 23, 25]),
(last_update:DATETIME, Examples: [2006-02-15 05:05:03.0])
]
# Table: payment
[
(payment_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(customer_id:INTEGER, Examples: [1, 2, 3]),
(staff_id:INTEGER, Examples: [1, 2]),
(rental_id:INTEGER, Examples: [76, 573, 1185]),
(amount:REAL, Examples: [2.99, 0.99, 5.99]),
(payment_date:DATETIME, Examples: [2005-05-25 11:30:37.0]),
(last_update:DATETIME, Examples: [2006-02-15 22:12:30.0])
]
# Table: film_text
[
(film_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(title:TEXT, Examples: [ACADEMY DINOSAUR, ACE GOLDFINGER, ADAPTATION HOLES]),
(description:TEXT)
]
# Table: customer
[
(customer_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(store_id:INTEGER, Examples: [1, 2]),
(first_name:TEXT, Examples: [MARY, PATRICIA, LINDA]),
(last_name:TEXT, Examples: [SMITH, JOHNSON, WILLIAMS]),
(email:TEXT),
(address_id:INTEGER, Examples: [5, 6, 7]),
(active:INTEGER, Examples: [1, 0]),
(create_date:DATETIME, Examples: [2006-02-14 22:04:36.0]),
(last_update:DATETIME, Examples: [2006-02-15 04:57:20.0])
]
# Table: staff
[
(staff_id:INTEGER, Primary Key, Examples: [1, 2]),
(first_name:TEXT, Examples: [Mike, Jon]),
(last_name:TEXT, Examples: [Hillyer, Stephens]),
(address_id:INTEGER, Examples: [3, 4]),
(picture:BLOB),
(email:TEXT),
(store_id:INTEGER, Examples: [1, 2]),
(active:INTEGER, Examples: [1]),
(username:TEXT, Examples: [Mike, Jon]),
(password:TEXT),
(last_update:DATETIME, Examples: [2006-02-15 04:57:16.0])
]
# Table: language
[
(language_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(name:TEXT, Examples: [English, Italian, Japanese]),
(last_update:DATETIME, Examples: [2006-02-15 05:02:19.0])
]
# Table: film_category
[
(film_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(category_id:INTEGER, Primary Key, Examples: [6, 11, 8]),
(last_update:DATETIME, Examples: [2006-02-15 05:07:09.0])
]
# Table: category
[
(category_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(name:TEXT, Examples: [Action, Animation, Children]),
(last_update:DATETIME, Examples: [2006-02-15 04:46:27.0])
]
# Table: actor
[
(actor_id:INTEGER, Primary Key, Examples: [1, 2, 3]),
(first_name:TEXT, Examples: [PENELOPE, NICK, ED]),
(last_name:TEXT, Examples: [GUINESS, WAHLBERG, CHASE]),
(last_update:DATETIME, Examples: [2006-02-15 04:34:33.0])
]
【Foreign keys】
film.original_language_id=language.language_id
film.language_id=language.language_id
rental.staff_id=staff.staff_id
rental.customer_id=customer.customer_id
rental.inventory_id=inventory.inventory_id
store.address_id=address.address_id
store.manager_staff_id=staff.staff_id
inventory.store_id=store.store_id
inventory.film_id=film.film_id
address.city_id=city.city_id
city.country_id=country.country_id
film_actor.film_id=film.film_id
film_actor.actor_id=actor.actor_id
payment.rental_id=rental.rental_id
payment.staff_id=staff.staff_id
payment.customer_id=customer.customer_id
customer.address_id=address.address_id
customer.store_id=store.store_id
staff.store_id=store.store_id
staff.address_id=address.address_id
film_category.category_id=category.category_id
film_category.film_id=film.film_id"""

        test_cases = [
            {
                "schema": movie_3_schema,
                "question": "Among the times Mary Smith had rented a movie, how many of them happened in June, 2005?",
                "evidence": "in June 2005 refers to year(payment_date) = 2005 and month(payment_date) = 6"
            },
            {
                "schema": movie_3_schema,
                "question": "Please give the full name of the customer who had made the biggest amount of payment in one single film rental.",
                "evidence": "full name refers to first_name, last_name; the biggest amount refers to max(amount)"
            },
            {
                "schema": movie_3_schema,
                "question": "How much in total had the customers in Italy spent on film rentals?",
                "evidence": "total = sum(amount); Italy refers to country = 'Italy'"
            },
            {
                "schema": movie_3_schema,
                "question": "Among the payments made by Mary Smith, how many of them are over 4.99?",
                "evidence": "over 4.99 refers to amount > 4.99"
            },
            {
                "schema": movie_3_schema,
                "question": "What is the average amount of money spent by a customer in Italy on a single film rental?",
                "evidence": "Italy refers to country = 'Italy'; average amount = divide(sum(amount), count(customer_id)) where country = 'Italy'"
            }
        ]
        
        print("="*80)
        print("🧪 QUICK INFERENCE TEST")
        print("="*80)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"Test Case {i}:")
            print(f"{'='*80}")
            print(f"\n📝 Question: {test_case['question']}")
            print(f"\n📊 Schema:")
            for line in test_case['schema'].split('\n')[:5]:  # Show first 5 lines
                print(f"   {line}")
            print("   ...")
            
            # Build prompt using the same template as training
            prompt_text = NL2SQLITE_TEMPLATE_EN.format(
                question=test_case['question'],
                db_schema=test_case['schema'],
                evidence=test_case['evidence']
            )
            
            # Create conversation format (same as training data)
            conversations = [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ]
            
            # Apply chat template (same as sql_infer.py)
            text = tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate (using similar params as sql_infer.py)
            print("\n⏳ Generating SQL...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Extract only the generated part (after input)
            generated_ids = outputs[0][len(inputs['input_ids'][0]):]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract only SQL, removing any explanations
            sql_output = extract_sql_only(raw_output)
            
            print(f"\n✅ Generated SQL:")
            print("─"*80)
            print(sql_output)
            print("─"*80)
            
            # Show raw output if it differs significantly (for debugging)
            if raw_output != sql_output and len(raw_output) > len(sql_output) + 20:
                print(f"\n⚠️  Note: Model also generated explanatory text (removed)")
                print(f"   Raw output length: {len(raw_output)} chars, SQL length: {len(sql_output)} chars")
        
        print(f"\n{'='*80}")
        print("✅ Quick inference test completed!")
        print("="*80)
        
        # Clean up to free memory
        print("\n🧹 Cleaning up memory...")
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        if 'base_model' in locals():
            del base_model
        torch.cuda.empty_cache()
        print("✅ Memory freed")
        
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick inference test for XiYan-SQL models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model (base or merged)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to the LoRA adapter (optional)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        exit(1)
        
    if args.adapter_path and not os.path.exists(args.adapter_path):
        print(f"❌ Adapter path does not exist: {args.adapter_path}")
        exit(1)
        
    run_quick_test(args.model_path, args.adapter_path)
