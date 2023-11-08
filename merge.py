from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

import os
import argparse


def main():

    device_arg = { 'device_map': 'auto' }

    base_model_name_or_path = r'C:\Users\Administrator\Desktop\git\llama2_chat_7b\llama_recipes\chat_7b_hf'
    peft_model_path = r'C:\Users\Administrator\Desktop\git\llama2_chat_7b\llama_recipes\PEFT\model'
    
    output_dir = r'C:\Users\Administrator\Desktop\git\llama2_chat_7b\llama_recipes\new_model'

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        return_dict=True,
        torch_dtype=torch.float16,
        **device_arg)

    model = PeftModel.from_pretrained(base_model, peft_model_path, **device_arg)

    model = model.merge_and_unload()


    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)


    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__" :
    main()