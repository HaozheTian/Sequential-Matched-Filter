import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from network import ECG_BiRNN, ECG_UNet

def normalize(x):
    min_val, max_val = x.min(), x.max()
    scale, bias = (max_val - min_val) / 2, (max_val + min_val) / 2
    return (x - bias) / scale

class ECGData(Dataset):
    def __init__(self, test=False):
        signal_folder = os.path.join('data', 'signal')
        peak_folder = os.path.join('data', 'peak')
        with os.scandir(peak_folder) as it:
            data_files = [entry.name for entry in it if entry.is_file()]
        data_files = sorted(data_files, key=lambda x: int(x.split('.')[0]))
        split_index = int(len(data_files) * 0.7)
        if test:
            data_file_names = data_files[split_index:]
        else:
            data_file_names = data_files[:split_index]

        self.features = torch.zeros(len(data_file_names), 250, dtype=torch.float32)
        self.labels = torch.zeros(len(data_file_names), 250, dtype=torch.float32)
        for idx, data_file_name in enumerate(data_file_names):
            ecg = np.load(os.path.join(signal_folder, data_file_name)).astype(np.float32)
            peak_inds = np.load(os.path.join(peak_folder, data_file_name)).astype(np.int16)
            feature = normalize(ecg)
            label = np.zeros(250, dtype=np.float32)
            label[peak_inds - 1] = 1
            self.features[idx] = torch.tensor(feature, dtype=torch.float32).reshape(1, -1)
            self.labels[idx] = torch.tensor(label, dtype=torch.float32).reshape(1, -1)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx, :].reshape(1, -1), self.labels[idx, :].reshape(1, -1)
    


def calc_metrics(preds, peaks):
    preds = torch.nonzero(preds).cpu().numpy().flatten()  # Convert tensor to numpy array
    peaks = torch.nonzero(peaks).cpu().numpy().flatten()  # Convert tensor to numpy array
    
    TP = 0
    matched = set()
    for pred in preds:
        close_peaks = np.where(np.abs(peaks - pred) < 5)[0]
        if len(close_peaks) > 0:
            closest_peak = peaks[close_peaks[0]]
            if closest_peak not in matched:
                TP += 1
                matched.add(closest_peak)

    FP = len(preds) - TP
    FN = len(peaks) - TP
    return TP, FN, FP

def validate(net, test_set):
    net.eval()  # Set model to evaluation mode
    TPs, FPs, FNs = 0, 0, 0
    with torch.no_grad():
        for feature, label in test_set:
            feature, label = feature.to(device), label.to(device)  # Move data to the device
            preds = net(feature.unsqueeze(0)).flatten()
            preds = preds / preds.max()
            preds = (preds > 0.5).float()
            peaks = label.flatten()
            TP, FP, FN = calc_metrics(preds, peaks)
            TPs += TP
            FPs += FP
            FNs += FN
    precision = TPs / (TPs + FPs) if (TPs + FPs) > 0 else 0.0
    recall = TPs / (TPs + FNs) if (TPs + FNs) > 0 else 0.0
    F1 = (2 * TPs) / (2 * TPs + FPs + FNs) if (2 * TPs + FPs + FNs) > 0 else 0.0
    return F1, precision, recall

for _ in range(5):
    # Set model:
    # model = 'UNet'
    model = 'BiRNN'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
    writer = SummaryWriter(log_dir='runs/' + model + '_'+'SMF'+time_str)

    num_epochs = 1000
    batch_size = 100
    learning_rate = 0.005
    val_freq = 50

    # Initialize network and move to device
    if model == 'UNet':
        net = ECG_UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
    elif model == 'BiRNN':
        net = ECG_BiRNN(hidden_size=64, num_layers=2).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load datasets
    train_dataset = ECGData(test=False)
    test_dataset = ECGData(test=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_F1, best_precision, best_recall = 0, 0, 0
    # Training loop
    with tqdm(range(num_epochs), desc="Training Progress") as pbar:
        num_step = 0
        for epoch in pbar:
            net.train()  # Set model to training mode
            running_loss = 0.0
            for batch_idx, (features, labels) in enumerate(train_loader):
                features, labels = features.to(device), labels.to(device)  # Move data to the device

                preds = net(features)
                optimizer.zero_grad()
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_step += 1
                pbar.set_postfix({"Loss": f"{running_loss/num_step:.4f}"})
            
            if epoch % val_freq == 0:
                F1, precision, recall = validate(net, test_dataset)
                writer.add_scalar('charts/precision', precision, num_step)
                writer.add_scalar('charts/recall', recall, num_step)
                writer.add_scalar('charts/F1', F1, num_step)
                print(precision, recall, F1)
                
                if F1 > best_F1:
                    best_F1, best_precision, best_recall = F1, precision, recall
                
    print(f'BEST: precsion={best_precision:.4f} recall={best_recall:.4f} F1={best_F1:.4f}')