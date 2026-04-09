import copy
import os

from datasets import load_from_disk

TRAIN_PATH = "./llm_distillation/datasets/hf/processed/pku_saferlhf/train"
TEST_PATH = "./llm_distillation/datasets/hf/processed/pku_saferlhf/test"

import os

from datasets import load_from_disk

TRAIN_PATH = "./llm_distillation/datasets/hf/processed/pku_saferlhf/train"
TEST_PATH = "./llm_distillation/datasets/hf/processed/pku_saferlhf/test"


def get_split(dataset_config, tokenizer, split):
    path = TRAIN_PATH if split == "train" else TEST_PATH
    raw = load_from_disk(path)
    max_len = getattr(dataset_config, "max_words", 1024)

    def tokenize(sample):
        chosen = "response_0" if sample["better_response_id"] == 0 else "response_1"

        prompt_text = f"Human: {sample['prompt']}\n\nAssistant: "
        full_text = prompt_text + sample[chosen] + tokenizer.eos_token

        tok = tokenizer(
            full_text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors=None,
        )

        labels = list(tok["input_ids"])

        # 3. Mask the prompt so loss is only calculated on the response
        # We tokenize the prompt separately to find its length
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), max_len)

        for i in range(prompt_len):
            labels[i] = -100

        # 4. Mask the padding tokens
        labels = [
            label if label != tokenizer.pad_token_id else -100 for label in labels
        ]

        tok["labels"] = labels
        return tok

    return raw.map(
        tokenize,
        remove_columns=raw.column_names,
        desc=f"Tokenizing {split}",
    )


# alias expected by data_utils.py
# get_split = get_custom_dataset
