from torch.utils.data import DataLoader, Dataset
import os
import glob
import torch
import torchaudio


class SoundDataset(Dataset):
    def __init__(self, sound_dir):
        self.sound_dir = sound_dir
        self.classes = {
            x: i for i, x in enumerate(os.listdir(sound_dir))
        }

        self.files = glob.glob(os.path.join(
            self.sound_dir, "**/*.wav"), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        class_name = os.path.basename(os.path.dirname(self.files[idx]))
        label = self.classes[class_name]
        waveform, _ = torchaudio.load(self.files[idx])
        return waveform, label


def collate_fn(wavs):
    max_len = max([w.shape[-1] for w in wavs])
    features = torch.zeros((len(wavs), 1, max_len))

    for i, wav in enumerate(wavs):
        features[i, 0, :wav.shape[-1]] = wav

    return features


def load_dataset(sound_dir, batch_size=2):
    train_set = SoundDataset(f"{sound_dir}/train")
    test_set = SoundDataset(f"{sound_dir}/test")

    train_dataset = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn)
    test_dataset = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataset, test_dataset
