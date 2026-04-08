import os
import copy
from datasets import load_from_disk

TRAIN_PATH = "./llm_distillation/datasets/hf/processed/pku_saferlhf/train"
TEST_PATH  = "./llm_distillation/datasets/hf/processed/pku_saferlhf/test"

def get_custom_dataset(dataset_config, tokenizer, split):
    path = TRAIN_PATH if split == "train" else TEST_PATH
    raw  = load_from_disk(path)
    max_len = getattr(dataset_config, "max_words", 1024)

    def tokenize(sample):
        chosen = "response_0" if sample["better_response_id"] == 0 else "response_1"
        text = (
            f"Human: {sample['prompt']}\n\nAssistant: {sample[chosen]}"
            + tokenizer.eos_token
        )
        tok = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors=None,
        )
        tok["labels"] = copy.deepcopy(tok["input_ids"])
        return tok

    return raw.map(
        tokenize,
        remove_columns=raw.column_names,
        desc=f"Tokenizing {split}",
    )

# alias expected by data_utils.py
get_split = get_custom_dataset