import torch
import torchaudio

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise


class SoundDataset(Dataset):
    def __init__(self, sound_dir, transforms=None):
        self.sound_dir = sound_dir
        self.classes = {
            x.name : i for i, x in enumerate(sound_dir.iterdir())
        }

        self.files = []
        self.transforms = transforms

        for ext in ("*.wav", "*.mp3"):
            self.files.extend(Path(self.sound_dir).rglob(ext))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        class_name = Path(self.files[idx]).parent.name
        label = self.classes[class_name]
        waveform, _ = torchaudio.load(self.files[idx])

        # Convert to mono
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, axis=0, keepdim=True)

        # Run augmentations
        if self.transforms:
            waveform = self.transforms(waveform)

        return waveform, label


def collate_fn(wavs):
    max_len = max([w[0].shape[-1] for w in wavs])

    labels = torch.LongTensor([w[1] for w in wavs])
    features = torch.zeros((len(wavs), 1, max_len))

    for i, (wav, _) in enumerate(wavs):
        features[i, 0, :wav.shape[-1]] = wav

    return features, labels


def load_dataset(sound_dir, batch_size=2, train_ratio=0.8):
    augmentations = Compose([
        TimeStretch(0.81, 1.23, p=0.5),
        PitchShift(-2, 2, p=0.5),
        AddGaussianNoise(0.001, 0.015, p=0.5),
    ])

    if list(Path(sound_dir).iterdir()) == ["test", "train"]:
        train_set = SoundDataset(Path(sound_dir, "train"), augmentations)
        test_set = SoundDataset(Path(sound_dir, "test"))
    else:
        dataset = SoundDataset(Path(sound_dir))
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

    train_dataset = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    test_dataset = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)

    return train_dataset, test_dataset
