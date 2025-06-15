import os
from pathlib import Path


class Config:
    # 路径配置
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = BASE_DIR / "models"

    # 数据文件
    TRAIN_FILE = RAW_DATA_DIR / "train.json"
    TEST_FILE = RAW_DATA_DIR / "test1.json"

    # 模型参数
    MODEL_NAME = "bert-base-chinese"
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    NUM_LABELS = 5  # 5种仇恨类型

    # 训练参数
    SEED = 42
    EARLY_STOPPING_PATIENCE = 3

    # 评估参数
    SOFT_MATCH_THRESHOLD = 0.5  # 软匹配阈值

    # 新增参数
    HATEFUL_THRESHOLD = 0.3  # hateful预测阈值调低
    CLASS_WEIGHTS = [1.0, 2.0, 2.0, 2.0, 2.0]  # 类别权重，加大仇恨类别的权重


config = Config()