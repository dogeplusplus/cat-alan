import torch
import torch.nn as nn
import torch.optim as optim

from models.m5 import M5
from preprocessing.pipeline import load_dataset


def main(sound_dir, model, epochs=5):
    train_ds, test_ds = load_dataset(sound_dir)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    size = len(train_ds)
    for e in range(epochs):
        for i, batch in enumerate(train_ds):
            import pdb
            pdb.set_trace()
            predictions = model(batch)
            loss = loss_fn(predictions, torch.zeros_like(predictions))

            optimizer.zero_grad()
            loss.backwards()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(),  i * len(batch)
                print(f"Loss: {loss:>7f}, [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = M5()
    model.to(device)

    sound_dir = "data/cats_dogs"
    main(sound_dir, model)
