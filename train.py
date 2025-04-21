import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from config.config import DataConfig, TrainConfig, CTCConfig
from data.dataset import SpectrogramDataset
from model.crnn import CRNN
from training.trainer import Trainer
from utils.helpers import set_seed, collate_fn


def main():
    set_seed(TrainConfig.RANDOM_STATE)

    symbols = [DataConfig.BLANK_CHAR] + list(DataConfig.SYMBOLS)
    char_to_index = {char: idx for idx, char in enumerate(symbols)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = pd.read_csv(DataConfig.CSV_FILE)
    train_df = full_df.iloc[:DataConfig.TRAIN_SIZE]
    test_df = full_df.iloc[DataConfig.TRAIN_SIZE:DataConfig.TRAIN_SIZE + DataConfig.TEST_SIZE] if DataConfig.TEST_SIZE > 0 else None

    train_dataset = SpectrogramDataset(train_df, DataConfig.ROOT_DIR, char_to_index, augment=True)
    test_dataset = SpectrogramDataset(test_df, DataConfig.ROOT_DIR, char_to_index, augment=False) if DataConfig.TEST_SIZE > 0 else None

    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)  # Аугментации отключены ради скорости обучения
    test_loader = DataLoader(test_dataset, batch_size=TrainConfig.BATCH_SIZE, collate_fn=collate_fn) if DataConfig.TEST_SIZE > 0 else None

    model = CRNN(num_classes=len(symbols)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=TrainConfig.LEARNING_RATE, weight_decay=TrainConfig.WEIGHT_DECAY)  # Говорят регуляризация в AdamW более стабильная
    criterion = nn.CTCLoss(blank=CTCConfig.BLANK_INDEX)

    trainer = Trainer(model, optimizer, criterion, device, symbols, test_loader)

    for epoch in range(TrainConfig.NUM_EPOCHS):
        avg_loss, avg_lev = trainer.train_epoch(train_loader, epoch, TrainConfig.NUM_EPOCHS)
        print(f"Эпоха [{epoch+1}/{TrainConfig.NUM_EPOCHS}] | "
              f"Средний лосс: {avg_loss:.4f} | Среднее Левенштейна: {avg_lev:.2f}")

        trainer.save_checkpoint(epoch, avg_loss)


if __name__ == "__main__":
    main()
