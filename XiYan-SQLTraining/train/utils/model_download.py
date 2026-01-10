#!/usr/bin/env python3
"""
Model Download Script for XiYan-SQL Training

Downloads Qwen2.5-Coder models from ModelScope or HuggingFace.
Supports interactive selection or command-line arguments.
"""

import sys
import os

def download_from_modelscope(model_name, cache_dir='../model/'):
    """Download model from ModelScope (faster in China)"""
    print(f"\n📥 Downloading from ModelScope: {model_name}")
    from modelscope import snapshot_download
    model_dir = snapshot_download(model_name, cache_dir=cache_dir)
    print(f"✅ Model downloaded to: {model_dir}")
    return model_dir

def download_from_huggingface(model_name, cache_dir='../model/'):
    """Download model from HuggingFace"""
    print(f"\n📥 Downloading from HuggingFace: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Downloading tokenizer...")
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Downloading model...")
    AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    print(f"✅ Model downloaded to: {cache_dir}")
    return cache_dir

def main():
    # Available models
    models = {
        '1': ('Qwen/Qwen2.5-Coder-0.5B-Instruct', '0.5B - Testing/Development'),
        '2': ('Qwen/Qwen2.5-Coder-1.5B-Instruct', '1.5B - Small deployment'),
        '3': ('Qwen/Qwen2.5-Coder-3B-Instruct', '3B - Recommended for XiYan-SQL'),
        '4': ('Qwen/Qwen2.5-Coder-7B-Instruct', '7B - Higher accuracy'),
        '5': ('Qwen/Qwen2.5-Coder-14B-Instruct', '14B - Best single model'),
        '6': ('Qwen/Qwen2.5-Coder-32B-Instruct', '32B - Maximum capability'),
    }

    # Check for command-line argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        source = sys.argv[2] if len(sys.argv) > 2 else 'modelscope'
    else:
        # Interactive selection
        print("=" * 60)
        print("XiYan-SQL Model Download")
        print("=" * 60)
        print("\nAvailable models:")
        for key, (name, desc) in models.items():
            print(f"  {key}. {name}")
            print(f"     {desc}")
        print()

        choice = input("Select model (1-6) [3]: ").strip() or '3'

        if choice not in models:
            print(f"❌ Invalid choice: {choice}")
            sys.exit(1)

        model_name = models[choice][0]

        print("\nDownload source:")
        print("  1. ModelScope (faster in China)")
        print("  2. HuggingFace (global)")
        source_choice = input("Select source (1-2) [1]: ").strip() or '1'
        source = 'modelscope' if source_choice == '1' else 'huggingface'

    # Download
    try:
        if source == 'modelscope':
            download_from_modelscope(model_name)
        else:
            download_from_huggingface(model_name)

        print("\n✅ Download complete!")
        print(f"\nYou can now train with this model by setting in xiyan_sft_3b.sh:")
        print(f'  MODEL="model/Qwen/{model_name.split("/")[1]}"')

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Try the other download source")
        print("  3. Install required packages:")
        print("     - ModelScope: pip install modelscope")
        print("     - HuggingFace: pip install transformers")
        sys.exit(1)

if __name__ == '__main__':
    main()













