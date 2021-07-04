import os
from torch.utils.data import DataLoader, Dataset

class SoundDataset(Dataset):
    def __init__(self, sound_dir):
        self.sound_dir = sound_dir


    def __len__(self):
        sound_files = sum([len(files) for r, d, files in os.walk(self.sound_dir)])
        return sound_files

    def __getitem__(self, idx):
        pass


def prepare_dataset():
    pass

ds = Dataset()
