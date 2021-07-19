import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from models.m5 import M5
from preprocessing.pipeline import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(sound_dir, model, epochs=5, print_every=5,
         save_every=5, batch_size=2):
    train_ds, test_ds = load_dataset(sound_dir, batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    bar_format = "{l_bar}{bar:20}{r_bar}{bar:-20b}"

    for e in range(epochs):
        # Train loop
        running_loss = 0
        running_correct = 0
        samples = 0
        train_bar = tqdm.tqdm(
            train_ds,
            desc=f"Train Epoch {e}",
            bar_format=bar_format
        )
        for i, batch in enumerate(train_bar):
            wavs, labels = batch
            wavs, labels = wavs.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(wavs)
            class_predictions = torch.squeeze(
                torch.argmax(predictions, dim=-1))
            loss = loss_fn(predictions[:, 0], labels)

            loss.backward()
            optimizer.step()
            running_correct += (class_predictions == labels).float().sum()
            running_loss += loss
            samples += labels.shape[0]

            if i % print_every == 0:
                train_bar.set_postfix(dict(
                    loss=f"{running_loss/samples:>7f}",
                    accuracy=f"{running_correct/samples:>7f}",
                ))

        mlflow.log_metric("train_loss", float(running_loss) / samples, step=e)
        mlflow.log_metric("train_accuracy", float(running_correct) / samples)

        # Test loop
        with torch.no_grad():
            running_loss = 0
            samples = 0
            running_correct = 0
            test_bar = tqdm.tqdm(
                test_ds,
                desc=f"Test  Epoch {e}",
                bar_format=bar_format
            )
            for i, batch in enumerate(test_bar):
                wavs, labels = batch
                wavs, labels = wavs.to(device), labels.to(device)

                predictions = model(wavs)
                class_predictions = torch.squeeze(
                    torch.argmax(predictions, dim=-1))
                loss = loss_fn(predictions[:, 0], labels)
                running_correct += (class_predictions == labels).float().sum()
                running_loss += loss
                samples += labels.shape[0]

                if i % print_every == 0:
                    test_bar.set_postfix(dict(
                        loss=f"{running_loss/samples:>7f}",
                        accuracy=f"{running_correct/samples:>7f}"
                    ))

        mlflow.log_metric("test_loss", float(running_loss) / samples, step=e)
        mlflow.log_metric("test_accuracy", float(running_correct) / samples)

        if (e + 1) % save_every == 0:
            mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    # sound_dir = "data/cats_dogs"
    sound_dir = "data/CatSound"

    with mlflow.start_run() as run:
        model = M5(filters=[64, 64, 128, 128], num_classes=10)
        model.to(device)

        main(sound_dir, model, epochs=100, batch_size=4)
