"""
Custom dataset module for MixLoRA training on choice question datasets.
Handles datasets where models need to predict choice answers (1, 2, 3, 4, etc.)
"""

import json
import logging
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer


class ChoiceQuestionDataset(Dataset):
    """
    Dataset class for choice question format data.

    Expected data format:
    {
        "instruction": "### Question: ... ### Answer: ",
        "input": "",  # Usually empty
        "output": "1",  # The correct choice number
        "label": "1"   # Not used for training
    }
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        choice_range: Optional[List[str]] = None,
        prompt_template: str = "default"
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the JSON dataset file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            choice_range: List of valid choice answers (e.g., ["1", "2", "3", "4"])
            prompt_template: Template for formatting prompts
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.choice_range = choice_range

        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.jsonl'):
                self.data = [json.loads(line) for line in f]
            else:
                self.data = json.load(f)

        # Auto-detect choice range if not provided
        if self.choice_range is None:
            self._detect_choice_range()

        # Validate data
        self._validate_data()

        logging.info(f"Loaded {len(self.data)} examples from {data_path}")
        logging.info(f"Choice range: {self.choice_range}")

    def _detect_choice_range(self):
        """Auto-detect the range of choices from the dataset."""
        outputs = set()
        for item in self.data:
            outputs.add(item['output'])

        # Try to sort numerically if possible
        try:
            sorted_outputs = sorted(outputs, key=int)
            self.choice_range = [str(x) for x in sorted_outputs]
        except ValueError:
            # If not numeric, sort alphabetically
            self.choice_range = sorted(list(outputs))

        logging.info(f"Auto-detected choice range: {self.choice_range}")

    def _validate_data(self):
        """Validate that all outputs are in the expected choice range."""
        invalid_count = 0
        valid_data = []

        for item in self.data:
            if item['output'] in self.choice_range:
                valid_data.append(item)
            else:
                invalid_count += 1
                logging.warning(f"Invalid output '{item['output']}' not in choice range {self.choice_range}")

        self.data = valid_data
        if invalid_count > 0:
            logging.warning(f"Removed {invalid_count} invalid examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Combine instruction and input (input is usually empty)
        input_text = item['instruction']
        if item.get('input', '').strip():
            input_text = input_text + item['input']

        # The target is just the choice number
        target_text = item['output']

        # Create the full text for training (input + target)
        full_text = input_text + target_text

        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        # Tokenize input only to get the split point
        input_tokenized = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )

        input_length = input_tokenized['input_ids'].shape[1]

        # Create labels (only the target part should contribute to loss)
        labels = tokenized['input_ids'].clone()
        labels[0, :input_length] = -100  # Ignore input part in loss calculation

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'choice_answer': target_text
        }


class ChoiceQuestionCollator:
    """
    Data collator for choice question datasets.
    Handles padding and batching.
    """

    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract individual components
        input_ids = [f['input_ids'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        choice_answers = [f['choice_answer'] for f in features]

        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)

        # Ensure minimum length to avoid 0-dimensional tensors
        max_length = max(max_length, 1)

        # Pad to multiple if specified
        if self.pad_to_multiple_of > 0:
            max_length = ((max_length + self.pad_to_multiple_of - 1)
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        padded_labels = []

        for i in range(len(features)):
            # Ensure input_ids has at least 1 element
            current_input_ids = input_ids[i]
            if len(current_input_ids) == 0:
                # If somehow we get empty input_ids, add pad token
                current_input_ids = torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)

            # Pad input_ids
            pad_length = max_length - len(current_input_ids)
            if pad_length > 0:
                padded_ids = torch.cat([
                    current_input_ids,
                    torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=current_input_ids.dtype)
                ])
            else:
                padded_ids = current_input_ids
            padded_input_ids.append(padded_ids)

            # Ensure attention_mask matches
            current_attention_mask = attention_masks[i]
            if len(current_attention_mask) != len(input_ids[i]):
                # Fix attention mask length if mismatched
                current_attention_mask = torch.ones(len(input_ids[i]), dtype=torch.long)

            # Pad attention_mask
            if pad_length > 0:
                padded_mask = torch.cat([
                    current_attention_mask,
                    torch.zeros(pad_length, dtype=current_attention_mask.dtype)
                ])
            else:
                padded_mask = current_attention_mask
            padded_attention_masks.append(padded_mask)

            # Ensure labels matches
            current_labels = labels[i]
            if len(current_labels) != len(input_ids[i]):
                # Fix labels length if mismatched
                current_labels = torch.full((len(input_ids[i]),), -100, dtype=torch.long)

            # Pad labels
            if pad_length > 0:
                padded_label = torch.cat([
                    current_labels,
                    torch.full((pad_length,), -100, dtype=current_labels.dtype)
                ])
            else:
                padded_label = current_labels
            padded_labels.append(padded_label)

        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels),
            'choice_answers': choice_answers
        }


def load_choice_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    choice_range: Optional[List[str]] = None
) -> ChoiceQuestionDataset:
    """
    Convenience function to load a choice question dataset.

    Args:
        data_path: Path to the dataset file
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        choice_range: Valid choice range (auto-detected if None)

    Returns:
        ChoiceQuestionDataset instance
    """
    return ChoiceQuestionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        choice_range=choice_range
    )


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = [
        {
            "instruction": "### Question: Give me the answer from 1 to 4: In the Netherlands, which of the following is an unusual common public practice? 1. Tipping generously 2. Using deodorant 3. Talking loudly on the phone. 4. Cycle everywhere. This question is for a country or language that is Netherlands. You can only choose one option.\n ### Answer: ",
            "instruction_mask": "### Question: Give me the answer from 1 to 4: In the [MASK], which of the following is an unusual common public practice? 1. Tipping generously 2. Using deodorant 3. Talking loudly on the phone. 4. Cycle everywhere. This question is for a country or language that is [MASK]. You can only choose one option.\n ### Answer: ",
            "input": "",
            "output": "1",
            "label": "1"
        }
    ]

    # Save sample data for testing
    with open('/tmp/sample_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)

    print("Sample dataset created for testing.")