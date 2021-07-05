import torch.nn as nn


class M5(nn.Module):
    def __init__(self,
                 kernel_sizes=[80, 3, 3, 3],
                 filters=[128, 128, 256, 512],
                 strides=[4, 1, 1, 1]):

        super(M5, self).__init__()
        assert len(kernel_sizes) == len(
            filters), "Kernel sizes and filters must be same length"
        assert len(filters) == len(
            strides), "Filters and strides must be same length"

        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.strides = strides

        self.model = nn.Sequential(
            nn.Conv1d(1,
                      self.filters[0],
                      kernel_size=self.kernel_sizes[0],
                      stride=self.strides[0]),
            nn.BatchNorm1d(self.filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(self.filters[0],
                      self.filters[1],
                      kernel_size=self.kernel_sizes[1],
                      stride=self.strides[1]),
            nn.BatchNorm1d(self.filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(self.filters[1],
                      self.filters[2],
                      kernel_size=self.kernel_sizes[2],
                      stride=self.strides[2]),
            nn.BatchNorm1d(self.filters[2]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(self.filters[2],
                      self.filters[3],
                      kernel_size=self.kernel_sizes[3],
                      stride=self.strides[3]),
            nn.BatchNorm1d(self.filters[3]),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.model(x)
