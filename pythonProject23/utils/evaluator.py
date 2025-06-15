# utils/evaluator.py
from sklearn.metrics import f1_score
from typing import List, Dict
import numpy as np
from difflib import SequenceMatcher


class Evaluator:
    @staticmethod
    def calculate_f1(y_true: List[int], y_pred: List[int]) -> float:
        """计算F1分数"""
        return f1_score(y_true, y_pred, average='macro')

    @staticmethod
    def string_similarity(s1: str, s2: str) -> float:
        """计算两个字符串的相似度"""
        return SequenceMatcher(None, s1, s2).ratio()

    @staticmethod
    def hard_match(true_item: Dict, pred_item: Dict) -> bool:
        """硬匹配评估"""
        return (true_item['target'] == pred_item['target'] and
                true_item['argument'] == pred_item['argument'] and
                true_item['group'] == pred_item['group'] and
                true_item['hateful'] == pred_item['hateful'])

    @staticmethod
    def soft_match(true_item: Dict, pred_item: Dict, threshold: float = 0.5) -> bool:
        """软匹配评估"""
        if true_item['group'] != pred_item['group'] or true_item['hateful'] != pred_item['hateful']:
            return False

        target_sim = Evaluator.string_similarity(true_item['target'], pred_item['target'])
        argument_sim = Evaluator.string_similarity(true_item['argument'], pred_item['argument'])

        return target_sim >= threshold and argument_sim >= threshold

    @staticmethod
    def evaluate_predictions(true_data: List[Dict], pred_data: List[Dict]) -> Dict:
        """评估预测结果"""
        hard_match_correct = 0
        soft_match_correct = 0

        for true_item, pred_item in zip(true_data, pred_data):
            if Evaluator.hard_match(true_item, pred_item):
                hard_match_correct += 1

            if Evaluator.soft_match(true_item, pred_item, config.SOFT_MATCH_THRESHOLD):
                soft_match_correct += 1

        total = len(true_data)
        hard_f1 = hard_match_correct / total if total > 0 else 0
        soft_f1 = soft_match_correct / total if total > 0 else 0
        avg_f1 = (hard_f1 + soft_f1) / 2

        return {
            'hard_f1': hard_f1,
            'soft_f1': soft_f1,
            'avg_f1': avg_f1
        }