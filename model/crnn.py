import torch.nn as nn
import torch.nn.functional as F
from config.config import ModelConfig


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN часть: извлекаем признаки из входных спектрограмм
        self.cnn = nn.Sequential(
            # -> (batch_size, 1, freq_bins, time_steps)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Первый слой (1 канал (моно) -> 64)
            nn.BatchNorm2d(64),  # Нормализация
            nn.ReLU(),           # Ф-ция активации
            nn.Dropout(0.1),     # Dropout, первым слоям реже свойственно переобучение => регуляризация слабее
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Делаем пуллинг только по времени, чтобы сохранить информацию о частоте

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(256, ModelConfig.CNN_OUTPUT_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ModelConfig.CNN_OUTPUT_CHANNELS),
            nn.ReLU(),
            nn.Dropout(0.4)     # Последним слоям чаще свойственно переобучение => регуляризация сильнее
            # -> (batch_size, CNN_OUTPUT_CHANNELS, freq_bins, time_steps)
        )

        # RNN часть: захватываем временные зависимости между признаками
        self.rnn = nn.LSTM(
            input_size=ModelConfig.CNN_OUTPUT_CHANNELS,  # Размерность признаков на входе в LSTM
            hidden_size=ModelConfig.RNN_HIDDEN_SIZE,     # Количество скрытых нейронов
            num_layers=ModelConfig.NUM_RNN_LAYERS,       # Количество слоев LSTM
            bidirectional=True,                          # Двунаправленная LSTM (учитывает контекст до и после текущего момента)
            batch_first=True,                            # (batch, seq, features) вместо (seq, batch, features)
            dropout=0.5                                  # Dropout
        )

        # Полносвязный слой для преобразования выхода LSTM в классы
        self.fc = nn.Linear(ModelConfig.RNN_HIDDEN_SIZE * 2, num_classes)  # *2 из-за bidirectional

    def forward(self, x):
        x = self.cnn(x)

        # Формат ожидаемый LSTM: batch, sequence_len, features:
        # (B, C, F, T) → (B, C*F, T)
        # (B, C*F, T) → (B, T, C*F)
        x = x.flatten(2).permute(0, 2, 1)

        x, _ = self.rnn(x)

        x = self.fc(x)
        # Приводим к формату для CTC Loss: (T, B, num_classes)
        x = x.permute(1, 0, 2)

        # Применяем логарифм softmax по классовой оси (CTCLoss ожидает вход в виде логарифма вероятностей)
        return F.log_softmax(x, dim=2)
