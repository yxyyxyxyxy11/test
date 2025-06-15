import torch
import json
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils.data_loader import HateSpeechDataset
from utils.preprocessor import TextPreprocessor
from config import config
from tqdm import tqdm
from pathlib import Path
import re


class HateSpeechClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(HateSpeechClassifier, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(config.MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.3)
        self.intermediate = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = torch.nn.GELU()
        self.group_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.hateful_classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.intermediate(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        group_logits = self.group_classifier(pooled_output)
        hateful_logits = self.hateful_classifier(pooled_output)
        return {
            'group_logits': group_logits,
            'hateful_logits': hateful_logits
        }


def extract_target_and_argument(text, group):
    """改进的目标和论点提取方法"""
    # 根据不同的group类型使用不同的提取策略
    if group == 'Racism':
        # 种族歧视通常有明确的种族词汇
        race_keywords = ['黑人', '白人', '黄种人', '种族', '民族', '黑人', '白人']
        for kw in race_keywords:
            if kw in text:
                start = text.find(kw)
                target = kw
                argument = text[start + len(kw):].strip()
                return target, argument
    elif group == 'Sexism':
        # 性别歧视通常有针对性别词汇
        gender_keywords = ['女人', '男人', '女性', '男性', '娘们', '汉子']
        for kw in gender_keywords:
            if kw in text:
                start = text.find(kw)
                target = kw
                argument = text[start + len(kw):].strip()
                return target, argument

    # 默认处理：取前两个词作为target，其余作为argument
    words = re.findall(r'\w+', text)
    if len(words) > 1:
        target = ' '.join(words[:2])
        argument = ' '.join(words[2:])
    else:
        target = text[:10].strip()
        argument = text.strip()

    return target, argument


def predict():
    # 加载测试数据
    preprocessor = TextPreprocessor()
    test_data = preprocessor.load_and_process_data(config.TEST_FILE, is_train=False)

    # 加载模型和tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = config.MODEL_DIR / "best_model"

    # 检查模型文件是否存在
    if not (model_dir / "pytorch_model.bin").exists():
        raise FileNotFoundError(f"模型文件未找到: {model_dir}")

    # 加载模型配置
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        model_config = json.load(f)

    # 初始化模型结构
    model = HateSpeechClassifier(model_config["num_labels"])

    # 加载模型权重
    model.load_state_dict(torch.load(model_dir / "pytorch_model.bin", map_location=device))
    model = model.to(device)
    model.eval()

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    # 创建数据加载器
    test_dataset = HateSpeechDataset(test_data, tokenizer, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 预测
    predictions = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="预测中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']

            outputs = model(input_ids, attention_mask)

            # 处理group预测
            group_probs = torch.softmax(outputs['group_logits'], dim=1)
            _, group_preds = torch.max(group_probs, dim=1)

            # 处理hateful预测
            hateful_probs = torch.sigmoid(outputs['hateful_logits'].squeeze())
            hateful_preds = (hateful_probs > config.HATEFUL_THRESHOLD).long()

            for i, text in enumerate(texts):
                # 获取预测结果
                group_mapping = {
                    0: 'Sexism',
                    1: 'Racism',
                    2: 'Region',
                    3: 'LGBTQ',
                    4: 'Others'
                }
                group = group_mapping.get(group_preds[i].item(), 'Others')
                hateful = 'hate' if hateful_preds[i].item() == 1 else 'non-hate'

                # 如果预测为非仇恨，则group设为Others
                if hateful == 'non-hate':
                    group = 'Others'

                # 提取target和argument
                target, argument = extract_target_and_argument(text, group)

                # 确保target和argument不为空
                target = target if target else text[:10].strip()
                argument = argument if argument else text.strip()

                predictions.append(f"{target}|{argument}|{group}|{hateful}")

    # 生成提交文件
    output_lines = []

    # 确保有2000行预测结果
    for i in range(2000):
        if i < len(predictions):
            output_lines.append(predictions[i])
        else:
            output_lines.append("||Others|non-hate")  # 不足补空行

    # 添加第2001行空行
    output_lines.append("")

    # 保存结果
    submission_file = Path("submission.txt")
    with open(submission_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"预测结果已保存到 {submission_file.absolute()}")
    print(f"共生成 {len(output_lines) - 1} 行预测结果 + 1 行空行")


if __name__ == "__main__":
    predict()