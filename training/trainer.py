import os
import random
import torch
from config.config import TrainConfig
from model.metrics import levenshtein_distance
from training.evaluator import Evaluator


class Trainer:
    def __init__(self, model, optimizer, criterion, device, symbols, test_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.symbols = symbols
        self.test_loader = test_loader
        self.best_loss = float("inf")
        self.evaluator = Evaluator(model, criterion, device, symbols)

    def train_epoch(self, train_loader, epoch, num_epochs):
        self.model.train()
        epoch_loss = 0.0
        total_levenshtein = 0
        total_samples = 0

        for batch_idx, (specs, targets, target_lens) in enumerate(train_loader):
            specs = specs.to(self.device)
            targets = targets.to(self.device)
            target_lens = target_lens.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(specs)
            # Расчет длинны входов
            input_lens = torch.full(
                size=(outputs.size(1),),     # batch size
                fill_value=outputs.size(0),  # длина временной оси
                dtype=torch.long
            ).to(self.device)

            loss = self.criterion(outputs, targets, input_lens, target_lens)
            epoch_loss += loss.item()
            loss.backward()
            # Раньше была проблема с взрывающимися градиентами
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), TrainConfig.GRAD_CLIP_MAX_NORM)
            # Делаем градиентный клиппинг (ограничивем норму градиентов до GRAD_CLIP_MAX_NORM)
            self.optimizer.step()

            # Декодирование предсказаний и истинных значений
            pred_strings = self.evaluator.decode_predictions(outputs)
            true_strings = [
                "".join([self.symbols[idx] for idx in targets[i][:target_lens[i]].tolist()])
                for i in range(targets.size(0))
            ]

            # Считаем метрики
            batch_levenshtein = 0
            for pred, true in zip(pred_strings, true_strings):
                batch_levenshtein += levenshtein_distance(pred, true)

            total_levenshtein += batch_levenshtein
            total_samples += targets.size(0)

            # Выводим информацию каждые 10 батчей
            if batch_idx % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_lev = total_levenshtein / total_samples if total_samples > 0 else 0

                # Выбираем случайный пример из батча
                random_idx = random.randint(0, len(pred_strings) - 1)
                random_pred = pred_strings[random_idx]
                random_true = true_strings[random_idx]

                print(f"Эпоха [{epoch + 1}/{num_epochs}] | Батч [{batch_idx}] | "
                      f"Лосс: {avg_loss:.4f} | Среднее Левенштейна: {avg_lev:.2f}")
                print(f"Предикт: {random_pred} | Истина: {random_true}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_lev_epoch = total_levenshtein / total_samples if total_samples > 0 else 0

        # Проверка на тестовом наборе после каждой эпохи
        if self.test_loader is not None:
            test_loss, test_lev = self.evaluator.evaluate(self.test_loader)
            print(f"\nТестирование после эпохи {epoch + 1} | Лосс: {test_loss:.4f} | Среднее Левенштейна: {test_lev:.2f}\n")
            self.model.train()  # Возвращаем модель в режим обучения

        return avg_epoch_loss, avg_lev_epoch

    def save_checkpoint(self, epoch, avg_loss):
        os.makedirs(TrainConfig.CHECKPOINT_DIR, exist_ok=True)
        # Сохраняем параметры после каждой эпохи
        model_params_path = os.path.join(
            TrainConfig.CHECKPOINT_DIR,
            f"model_params_ep{epoch + 1}.pth"
        )
        torch.save(self.model.state_dict(), model_params_path)
        # Сохраняем параметры лучшей модели
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            best_model_path = os.path.join(TrainConfig.CHECKPOINT_DIR, "best_model.pth")
            torch.save(self.model.state_dict(), best_model_path)
