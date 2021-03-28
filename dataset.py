from torch.utils.data import Dataset
import pandas as pd
import torch


class SARCDataset(Dataset):
    def __init__(self, data, tokenizer, max_len: int = 200):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        :return
            label: 0(non-sarcastin) or 1(sarcastic)
            comment: str
        """
        self.data = data
        self.size = len(self.data)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.size

    def __repr__(self):
        return str(self.data.head())

    def __getitem__(self, idx):
        comment_text = self.data.iloc[idx]['comment']
        label = int(self.data.iloc[idx]['label'])
        encoding = self.tokenizer(comment_text, return_tensors='pt', padding=True, truncation=True)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label).long()
        }