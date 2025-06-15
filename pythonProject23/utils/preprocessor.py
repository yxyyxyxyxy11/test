# utils/preprocessor.py
import re
import jieba
from zhconv import convert
from typing import Dict, List, Tuple
import json
from config import config


class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        text = convert(text, 'zh-cn')  # 繁体转简体
        text = re.sub(r'[^\w\s]', '', text)  # 去除标点
        text = re.sub(r'\s+', ' ', text)  # 去除多余空格
        return text.strip()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """分词"""
        return list(jieba.cut(text))

    @staticmethod
    def load_and_process_data(file_path: str, is_train: bool = True) -> List[Dict]:
        """加载并处理数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            text = item.get('text', '')
            if is_train:
                target = item.get('target', '')
                argument = item.get('argument', '')
                group = item.get('group', '')
                hateful = item.get('hateful', False)

                # 清理文本
                cleaned_text = TextPreprocessor.clean_text(text)
                cleaned_target = TextPreprocessor.clean_text(target)
                cleaned_argument = TextPreprocessor.clean_text(argument)

                processed_data.append({
                    'text': cleaned_text,
                    'target': cleaned_target,
                    'argument': cleaned_argument,
                    'group': group,
                    'hateful': hateful
                })
            else:
                cleaned_text = TextPreprocessor.clean_text(text)
                processed_data.append({
                    'text': cleaned_text
                })

        return processed_data