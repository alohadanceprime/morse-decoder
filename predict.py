import pandas as pd
import torch
from torch.utils.data import DataLoader
from config.config import DataConfig, CTCConfig
from data.dataset import TestSpectrogramDataset
from model.crnn import CRNN
from utils.helpers import collate_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    symbols = [DataConfig.BLANK_CHAR] + list(DataConfig.SYMBOLS)

    model = CRNN(num_classes=len(symbols))
    model.load_state_dict(torch.load("model_params_ep40.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_df = pd.read_csv("test.csv")
    ids = test_df["id"].tolist()

    test_dataset = TestSpectrogramDataset(ids, root_dir=DataConfig.ROOT_DIR)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_test)

    all_ids = []
    all_messages = []

    with torch.no_grad():
        for specs, batch_ids in test_loader:
            specs = specs.to(device)
            outputs = model(specs)

            _, max_indices = torch.max(outputs, 2)
            for i, batch in enumerate(max_indices.permute(1, 0)):
                chars = []
                prev = None
                for idx in batch:
                    idx = idx.item()
                    if idx != CTCConfig.BLANK_INDEX and idx != prev:
                        chars.append(symbols[idx])
                    prev = idx
                all_messages.append("".join(chars))

            all_ids.extend(batch_ids)

    df = pd.DataFrame({"id": all_ids, "message": all_messages})
    df["id"] = df["id"].str.replace(r"\.npy$", ".opus", regex=True)
    df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
