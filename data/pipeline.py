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


if __name__ == "__main__":
    sound_dir = "/mnt/c/Users/Albert/Downloads/archive"
    ds = SoundDataset(sound_dir)

    data_generator = DataLoader(ds)

    lengths = []
    for x in data_generator:
        lengths.append(x.numpy().size)

    plt.hist(lengths)
    plt.show()
