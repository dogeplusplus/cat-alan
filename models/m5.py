import torch.nn as nn


class M5(nn.Module):
    def __init__(self,
                 kernel_sizes=[80, 3, 3, 3],
                 filters=[16, 16, 32, 64],
                 strides=[16, 1, 1, 1],
                 num_classes=2):

        super(M5, self).__init__()
        assert len(kernel_sizes) == len(
            filters), "Kernel sizes and filters must be same length"
        assert len(filters) == len(
            strides), "Filters and strides must be same length"

        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.strides = strides
        self.num_classes = num_classes

        self.model = nn.Sequential(
            nn.Conv1d(1,
                      self.filters[0],
                      kernel_size=self.kernel_sizes[0],
                      stride=self.strides[0]),
            nn.BatchNorm1d(self.filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
            nn.Conv1d(self.filters[0],
                      self.filters[1],
                      kernel_size=self.kernel_sizes[1],
                      stride=self.strides[1]),
            nn.BatchNorm1d(self.filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
            nn.Conv1d(self.filters[1],
                      self.filters[2],
                      kernel_size=self.kernel_sizes[2],
                      stride=self.strides[2]),
            nn.BatchNorm1d(self.filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Dropout(0.3),
            nn.Conv1d(self.filters[2],
                      self.filters[3],
                      kernel_size=self.kernel_sizes[3],
                      stride=self.strides[3]),
            nn.BatchNorm1d(self.filters[3]),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        self.fc = nn.Linear(self.filters[3], self.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.model(x)
        x = nn.AvgPool1d(x.shape[-1])(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x
