import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from models.m5 import M5
from preprocessing.pipeline import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(sound_dir, model, epochs=5, print_every=5):
    train_ds, test_ds = load_dataset(sound_dir)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for e in range(epochs):
        # Train loop
        running_loss = 0
        samples = 0
        train_bar = tqdm.tqdm(
            train_ds,
            desc=f"Train Epoch {e}",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        for i, batch in enumerate(train_bar):
            wavs, labels = batch
            wavs, labels = wavs.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(wavs)
            loss = loss_fn(predictions[:, 0], labels)
            loss.backward()
            optimizer.step()

            running_loss += loss
            samples += labels.shape[0]

            if i % print_every == 0:
                train_bar.set_postfix(dict(loss=f"{running_loss/samples:>7f}"))

        # Test loop
        with torch.no_grad():
            running_loss = 0
            samples = 0
            test_bar = tqdm.tqdm(
                test_ds,
                desc=f"Test  Epoch {e}",
                bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )
            for i, batch in enumerate(test_bar):
                wavs, labels = batch
                wavs, labels = wavs.to(device), labels.to(device)
                predictions = model(wavs)
                loss = loss_fn(predictions[:, 0], labels)

                running_loss += loss
                samples += labels.shape[0]

                if i % print_every == 0:
                    test_bar.set_postfix(
                        dict(loss=f"{running_loss/samples:>7f}"))


if __name__ == "__main__":

    model = M5()
    model.to(device)

    sound_dir = "data/cats_dogs"
    main(sound_dir, model, epochs=100)
