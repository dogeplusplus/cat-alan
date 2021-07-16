from torch.utils.data import DataLoader, Dataset, random_split
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

        self.files = []

        for ext in ("wav", "mp3"):
            self.files.extend(glob.glob(os.path.join(
                self.sound_dir, f"**/*.{ext}"), recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        class_name = os.path.basename(os.path.dirname(self.files[idx]))
        label = self.classes[class_name]
        waveform, _ = torchaudio.load(self.files[idx])

        # Convert to mono
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, axis=0, keepdim=True)
        return waveform, label


def collate_fn(wavs):
    max_len = max([w[0].shape[-1] for w in wavs])

    labels = torch.LongTensor([w[1] for w in wavs])
    features = torch.zeros((len(wavs), 1, max_len))

    for i, (wav, _) in enumerate(wavs):
        features[i, 0, :wav.shape[-1]] = wav

    return features, labels


def load_dataset(sound_dir, batch_size=2, train_ratio=0.8):
    if os.listdir(sound_dir) == ["test", "train"]:
        train_set = SoundDataset(f"{sound_dir}/train")
        test_set = SoundDataset(f"{sound_dir}/test")
    else:
        dataset = SoundDataset(sound_dir)
        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        train_set, test_set = random_split(dataset, [train_size, test_size])

    train_dataset = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn)
    test_dataset = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn)

    return train_dataset, test_dataset
