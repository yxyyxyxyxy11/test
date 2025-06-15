import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils.data_loader import HateSpeechDataset, create_data_loaders
from utils.preprocessor import TextPreprocessor
from utils.evaluator import Evaluator
from config import config
import os
import json
from tqdm import tqdm
import numpy as np
import random


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(config.SEED)


class HateSpeechClassifier(nn.Module):
    def __init__(self, num_labels):
        super(HateSpeechClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(config.MODEL_NAME)
        self.dropout = nn.Dropout(0.3)  # 增加dropout防止过拟合

        # 增加一个中间层
        self.intermediate = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.GELU()

        # 分类head
        self.group_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.hateful_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs[1]

        # 增加中间层处理
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.intermediate(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # 分类任务
        group_logits = self.group_classifier(pooled_output)
        hateful_logits = self.hateful_classifier(pooled_output)

        return {
            'group_logits': group_logits,
            'hateful_logits': hateful_logits
        }


def train_model():
    # 加载和预处理数据
    preprocessor = TextPreprocessor()
    train_data = preprocessor.load_and_process_data(config.TRAIN_FILE)

    # 划分训练集和验证集 (80-20)
    split_idx = int(0.8 * len(train_data))
    train_set = train_data[:split_idx]
    val_set = train_data[split_idx:]

    # 初始化tokenizer和数据加载器
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    train_loader, val_loader = create_data_loaders(train_set, val_set, tokenizer, config.BATCH_SIZE)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HateSpeechClassifier(config.NUM_LABELS).to(device)

    # 优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 添加类别权重
    group_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float).to(device)
    group_criterion = nn.CrossEntropyLoss(weight=group_weights)
    hateful_criterion = nn.BCEWithLogitsLoss()

    # 训练循环
    best_f1 = 0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            group_labels = batch['group'].to(device)
            hateful_labels = batch['hateful'].float().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # 计算group分类损失
            group_loss = group_criterion(outputs['group_logits'], group_labels)

            # 计算hateful分类损失
            hateful_loss = hateful_criterion(
                outputs['hateful_logits'].squeeze(),
                hateful_labels
            )

            # 总损失
            total_batch_loss = group_loss + hateful_loss
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # 验证
        val_metrics = evaluate(model, val_loader, device)
        print(f"Validation Metrics: {val_metrics}")

        # 早停和模型保存
        if val_metrics['avg_f1'] > best_f1:
            best_f1 = val_metrics['avg_f1']
            patience_counter = 0
            save_model(model, tokenizer)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break


def evaluate(model, data_loader, device):
    model.eval()
    group_preds = []
    group_labels = []
    hateful_preds = []
    hateful_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            group_labels_batch = batch['group'].cpu().numpy()
            hateful_labels_batch = batch['hateful'].cpu().numpy()

            outputs = model(input_ids, attention_mask)

            # 处理group预测
            _, group_pred = torch.max(outputs['group_logits'], dim=1)
            group_preds.extend(group_pred.cpu().numpy())
            group_labels.extend(group_labels_batch)

            # 处理hateful预测
            hateful_pred = torch.sigmoid(outputs['hateful_logits'].squeeze())
            hateful_pred = (hateful_pred > config.HATEFUL_THRESHOLD).long().cpu().numpy()
            hateful_preds.extend(hateful_pred)
            hateful_labels.extend(hateful_labels_batch)

    # 计算各项指标
    group_f1 = Evaluator.calculate_f1(group_labels, group_preds)
    hateful_f1 = Evaluator.calculate_f1(hateful_labels, hateful_preds)
    avg_f1 = (group_f1 + hateful_f1) / 2

    return {
        'group_f1': group_f1,
        'hateful_f1': hateful_f1,
        'avg_f1': avg_f1
    }


def save_model(model, tokenizer):
    output_dir = config.MODEL_DIR / "best_model"
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # 保存模型状态字典
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")

    # 保存tokenizer
    tokenizer.save_pretrained(output_dir)

    # 保存模型配置
    model_config = {
        "num_labels": config.NUM_LABELS,
        "model_name": config.MODEL_NAME
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(model_config, f)

    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    train_model()