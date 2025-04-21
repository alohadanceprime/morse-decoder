import torch
from model.metrics import levenshtein_distance
from config.config import CTCConfig


class Evaluator:
    def __init__(self, model, criterion, device, symbols):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.symbols = symbols

    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        total_levenshtein = 0
        total_samples = 0

        with torch.no_grad():  # TODO: Потраить inference mode?
            for batch_idx, (specs, targets, target_lens) in enumerate(test_loader):
                specs = specs.to(self.device)
                targets = targets.to(self.device)
                target_lens = target_lens.to(self.device)

                outputs = self.model(specs)
                input_lens = torch.full(
                    size=(outputs.size(1),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(self.device)

                loss = self.criterion(outputs, targets, input_lens, target_lens)
                total_loss += loss.item()

                pred_strings = self.decode_predictions(outputs)
                true_strings = [
                    "".join([self.symbols[idx] for idx in targets[i][:target_lens[i]].tolist()])
                    for i in range(targets.size(0))
                ]

                for pred, true in zip(pred_strings, true_strings):
                    total_levenshtein += levenshtein_distance(pred, true)
                    total_samples += 1

        return total_loss / len(test_loader), total_levenshtein / total_samples if total_samples > 0 else 0

    def decode_predictions(self, outputs):
        _, max_indices = torch.max(outputs, 2)
        decoded = []
        for batch in max_indices.permute(1, 0):
            chars = []
            prev = None
            for idx in batch:
                idx = idx.item()
                if idx != CTCConfig.BLANK_INDEX and idx != prev:  # Не декодируем BLANK_INDEX, а так же если на временном промежутке 2 одинаковых символа
                    chars.append(self.symbols[idx])
                prev = idx
            decoded.append("".join(chars))
        return decoded
