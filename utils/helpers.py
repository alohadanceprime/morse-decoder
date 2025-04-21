import random
import numpy as np
import torch
from config.config import CTCConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # TODO: поменять на false
    # Запрещает pytorch использовать недетерминированные алгоритмы, даже если те быстрее
    torch.backends.cudnn.benchmark = False  # TODO: поменять на true
    # Отключает автовыбор оптимального (по скорости) для заданного размера входных данных на основе бенчмарка
    # То есть всегда используется 1 и тот же алгоритм


def collate_fn(batch):
    specs, targets, target_lengths = zip(*batch)
    specs = torch.stack(specs)  # Объединяем спектрограммы в 1 батч
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=CTCConfig.BLANK_INDEX)
    # Дополняем таргеты до одинаковой длинны с помощью BLANK_INDEX
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return specs, targets, target_lengths


def collate_test(batch):
    specs = [item[0] for item in batch]
    ids = [item[1] for item in batch]
    specs = torch.stack(specs, dim=0)
    return specs, ids
