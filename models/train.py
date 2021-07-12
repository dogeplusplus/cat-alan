import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from models.m5 import M5
from preprocessing.pipeline import load_dataset


def main(sound_dir, model, epochs=5, print_every=5):
    train_ds, test_ds = load_dataset(sound_dir)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for e in range(epochs):
        pbar = tqdm.tqdm(train_ds, desc=f"Epoch {e}")
        for i, batch in enumerate(pbar):
            wavs, labels = batch
            optimizer.zero_grad()
            predictions = model(wavs)
            loss = loss_fn(torch.squeeze(predictions), labels)

            loss.backward()
            optimizer.step()

            if i % print_every == 0:
                pbar.set_postfix(dict(loss=f"{loss.item():>7f}"))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = M5()
    model.to(device)

    sound_dir = "data/cats_dogs"
    main(sound_dir, model)
