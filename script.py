import torchaudio
import torch
from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from models.m5 import M5

device = "cuda"


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip())
                        for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + \
                load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


train_set = SubsetSC("training")
test_set = SubsetSC("testing")
waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

labels = sorted(list(set(datapoint[2] for datapoint in train_set)))


def label_to_index(w):
    return torch.tensor(labels.index(w))


def index_to_label(i):
    return labels[i]


def pad_sequence(batch):
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def collate_fn(batch):
    tensors, targets = [], []

    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets


batch_size = 4
num_workers = 1
pin_memory = True


train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_fn,
    num_workers=num_workers,
    pin_memory=pin_memory,
)


def train(model, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        loss = F.nll_loss(output.squeeze(), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        # if batch_idx % log_interval == 0:

        # update progress bar
        # pbar.update(pbar_update)
        pbar.set_postfix(dict(loss=loss.item()))
        # record loss
        losses.append(loss.item())


model = M5(num_classes=35)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
# reduce the learning after 20 epochs by a factor of 10
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

log_interval = 20
n_epoch = 2

# pbar_update = 1 / (len(train_loader) + len(test_loader))
losses = []

new_sample_rate = 8000
transform = torchaudio.transforms.Resample(
    orig_freq=sample_rate, new_freq=new_sample_rate)
transform = transform.to(device)

with tqdm(total=n_epoch) as pbar:
    for epoch in range(1, n_epoch + 1):
        train(model, epoch, log_interval)
        scheduler.step()
