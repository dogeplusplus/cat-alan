from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
import glob
import torch
import torchaudio
import matplotlib
matplotlib.use("TkAgg")


class SoundDataset(Dataset):
    def __init__(self, sound_dir):
        self.sound_dir = sound_dir
        self.files = glob.glob(os.path.join(
            self.sound_dir, "**/*.wav"), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        waveform, sample_rate = torchaudio.load(self.files[idx])
        return waveform


def collate_fn(wavs):
    max_len = max([w.shape[-1] for w in wavs])
    features = torch.zeros((len(wavs), max_len))

    for i, wav in enumerate(wavs):
        features[i, :wav.shape[-1]] = wav

    return features


if __name__ == "__main__":
    sound_dir = "/mnt/c/Users/Albert/Downloads/archive"
    ds = SoundDataset(sound_dir)

    data_generator = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

    lengths = []
    for x in data_generator:
        lengths.append(x.numpy().size)

    plt.hist(lengths)
    plt.show()
