import random
import torch

def augment_spectrogram(spectrogram):
    # Аугментация сводит на нет переобучение, но с ней требуется слишком много эпох

    # С вероятностью 0.3 добавляем шум к спектрограмме
    if random.random() < 0.3:
        noise = torch.randn_like(spectrogram) * 0.05  # создаём шум с теми же размерами по закону нормального распределения
        spectrogram = spectrogram + noise

    # С вероятностью 0.3 применяем временную маскировку
    if random.random() < 0.3:
        time_dim = spectrogram.size(-1)  # определяем длину временного измерения
        t = random.randint(0, time_dim - 1)  # выбираем случайную позицию начала маски
        width = max(1, int(0.1 * time_dim))  # ширина маски — до 10% от длины временной оси
        # обнуляем значения в выбранной временной области
        spectrogram[..., t:min(t + width, time_dim)] = 0
    return spectrogram
