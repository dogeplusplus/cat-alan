import uuid
import tqdm
import json
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

from pathlib import Path
from models.m5 import M5
from preprocessing.pipeline import load_dataset

DEFAULTS = dict(
    model_name=uuid.uuid4(),
    kernel_sizes=[80, 3, 3, 3],
    filters=[64, 64, 128, 128],
    strides=[16, 1, 1, 1],
    epochs=10,
    batch_size=2,
)


def train(train_ds, test_ds, model, epochs, device,
          print_every=5, save_every=5):

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
        mlflow.log_metric("train_accuracy", float(running_correct) / samples, step=e)

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
        mlflow.log_metric("test_accuracy", float(running_correct) / samples, step=e)

        if (e + 1) % save_every == 0:
            mlflow.pytorch.log_model(model, "model")


def parse_arguments():
    parser = argparse.ArgumentParser("Train audio classification model.")
    parser.add_argument("-c", "--config", type=str,
                        help="Path to training config")

    args = parser.parse_args()
    with open(args.config) as f:
        args = json.load(f)

    return args


def main(args):
    args = parse_arguments()
    dataset_path = args["dataset_path"]

    for parameter in DEFAULTS.keys():
        if parameter not in args:
            args[parameter] = DEFAULTS[parameter]

    num_classes = 0
    for maybe_class in Path(dataset_path).iterdir():
        if maybe_class.is_dir():
            num_classes += 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with mlflow.start_run():
        model = M5(
            filters=args["filters"],
            kernel_sizes=args["kernel_sizes"],
            strides=args["strides"],
            num_classes=num_classes
        )
        mlflow.log_dict(args, "config.json")
        model.to(device)
        train_ds, test_ds = load_dataset(dataset_path, args["batch_size"])
        train(train_ds, test_ds, model, args["epochs"], device)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
