# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import datasets
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
)
import os
from datasets_processing.utils import Concatenator

def get_custom_dataset(dataset_config, tokenizer, split):


    data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'personal_dataset', 'format_data.json' )
    json_dataset = datasets.load_dataset(path='json', data_files=data_file)

    split_dataset = json_dataset["train"].train_test_split(test_size=0.01, seed=2357, shuffle=True)
    split_dataset['validation'] = split_dataset.pop('test') # rename the test split to val

    dataset = split_dataset[split]

    prompt = (
        f"Question:\n{{question}}\n---\n Answer:\n{{answer}}{{eos_token}}"
    )

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                question=sample["question"],
                answer=sample["answer"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)

    return dataset

if __name__ == "__main__":

    token_path = r'C:\Users\Administrator\Desktop\git\llama2_chat_7b\llama_recipes\chat_7b_hf'
    tokenizer = LlamaTokenizer.from_pretrained(token_path)
    print(len(tokenizer))
    # res = get_custom_dataset(1, tokenizer, split = 'train')
    # print(res)