from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import Dict, List
import torch
from config import config


class HateSpeechDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, is_train: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_length = config.MAX_LENGTH

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if self.is_train:
            # 将group映射为数字标签
            group_mapping = {
                'Sexism': 0,
                'Racism': 1,
                'Region': 2,
                'LGBTQ': 3,
                'Others': 4
            }
            group = group_mapping.get(item['group'], 4)

            # 处理hateful标签
            hateful = 1 if item['hateful'] else 0

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'text': text,
                'target': item.get('target', ''),
                'argument': item.get('argument', ''),
                'group': torch.tensor(group, dtype=torch.long),
                'hateful': torch.tensor(hateful, dtype=torch.long)
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'text': text
            }


def create_data_loaders(train_data, val_data, tokenizer, batch_size=32):
    train_dataset = HateSpeechDataset(train_data, tokenizer)
    val_dataset = HateSpeechDataset(val_data, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader