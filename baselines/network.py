import torch
import torch.nn as nn
import torch.nn.functional as F

class ECG_BiRNN(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        super(ECG_BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, bidirectional=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape (batch, 1, 250) → (batch, 250, 1)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)  # Ensure h0 is on the correct device
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return torch.sigmoid(out).permute(0, 2, 1)

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down1D(nn.Module):
    """Downsampling + double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up1D(nn.Module):
    """Upsampling + skip connection + double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = DoubleConv1D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])
        x = torch.cat([x2, x1], dim=1)  # Skip connection
        return self.conv(x)

class OutConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ECG_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ECG_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv1D(n_channels, 64)
        self.down1 = Down1D(64, 128)
        self.down2 = Down1D(128, 256)
        self.down3 = Down1D(256, 512)
        self.down4 = Down1D(512, 1024)

        factor = 2 if bilinear else 1
        self.up1 = Up1D(1024, 512 // factor, bilinear)
        self.up2 = Up1D(512, 256 // factor, bilinear)
        self.up3 = Up1D(256, 128 // factor, bilinear)
        self.up4 = Up1D(128, 64, bilinear)
        self.outc = OutConv1D(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)  # Skip connection
        x = self.up2(x, x3)   # Skip connection
        x = self.up3(x, x2)   # Skip connection
        x = self.up4(x, x1)   # Skip connection
        logits = self.outc(x)
        return torch.sigmoid(logits)