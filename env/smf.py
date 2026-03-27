import gymnasium
import os
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate, find_peaks




def reward_function_balanced(pks_preds, pks_gts, tolerance=5):
    # use broadcast if preds=[a, b, c, d], preds_correct=[True, False, True, True]
    preds_correct = np.any(np.abs(pks_preds.reshape((-1, 1)) - pks_gts.reshape((1, -1))) < tolerance, axis=1)
    num_preds, num_gts = len(pks_preds), len(pks_gts)
    TP = preds_correct.sum()
    FP = num_preds - TP
    FN = num_gts - TP

    TP = TP/num_gts if num_gts>0 else 0
    FN = FN/num_gts if num_gts>0 else 0
    FP = FP/num_preds if num_preds>0 else 0

    reward = 10 * TP - 5 * FP - 5 * FN
    return reward, TP, FP, FN


def reward_function_F1(pks_preds, pks_gts, tolerance=5):
    # Check for correct predictions
    preds_correct = np.any(np.abs(pks_preds.reshape((-1, 1)) - pks_gts.reshape((1, -1))) < tolerance, axis=1)
    num_preds, num_gts = len(pks_preds), len(pks_gts)
    TP = preds_correct.sum()
    FP = num_preds - TP
    FN = num_gts - TP

    reward = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    return reward, TP, FP, FN





class SMF:
    def __init__(self, ph= 0.5, pd = 30, **kwargs) -> None:
        self.render = kwargs.get('render', False)
        self.fs = kwargs.get('fs', 200)
        self.template_len = kwargs.get("template_len", 8)
        self.eps_len = kwargs.get("eps_len", 5)
        print(f'episode length is {self.eps_len}')
        self.tolerance = kwargs.get("tolerance", 5)  
        self.data_dir = kwargs.get('data_dir', 'data')
        self.f1_reward = kwargs.get('f1_reward', False)
        self.num_reset = 0
        self.ph = ph
        self.pd = pd

        dir = os.path.join(self.data_dir, 'peak')
        with os.scandir(dir) as it:
            data_files = [entry.name for entry in it if entry.is_file()]
        data_files = sorted(data_files, key=lambda x: int(x.split('.')[0]))
        split_index = int(len(data_files)*0.7)
        self.train_data_files, self.test_data_files = data_files[:split_index], data_files[split_index:]
        self.num_train, self.num_test = len(self.train_data_files), len(self.test_data_files)
        
        self.ecg_len = np.load(os.path.join('data', 'signal', data_files[0])).shape[0]
        self.observation_space = gymnasium.spaces.Box(
            low=np.concatenate((np.full((1, self.ecg_len), -1, dtype=np.float32), np.array([[0]], dtype=np.float32)), axis=1),
            high=np.concatenate((np.full((1, self.ecg_len), 1, dtype=np.float32), np.array([[self.eps_len - 1]], dtype=np.float32)), axis=1),
            dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.template_len,), dtype=np.float32)


    def reset(self, seed = 0, status = 'train', idx = None):
        self.len=0
        self._set_init_state(seed+self.num_reset, status, idx)
        obs = np.concatenate([self.state, [self.len]]).reshape(1, -1)

        if self.render:
            self.act_buffer, self.obs_buffer = [], [obs]
            self.preds_buffer = []

        self.num_reset += 1
        return obs, {}
    

    def step(self, act, status = 'train'):
        state = correlate(self.state, act, mode='same')
        state = self.normalize(state)
        preds, _ = find_peaks(state, height=self.ph, distance=self.pd)

        self.len += 1
        self.state = state
        obs = np.concatenate([self.state, [self.len]]).reshape(1, -1)
        if status == 'train':
            if self.f1_reward:
                rew, TP, FP, FN = reward_function_F1(pks_preds=preds, pks_gts=self.peaks, tolerance=self.tolerance)
            else:
                rew, TP, FP, FN = reward_function_balanced(pks_preds=preds, pks_gts=self.peaks, tolerance=self.tolerance)
        elif status == 'test':
            rew, TP, FP, FN = reward_function_F1(pks_preds=preds, pks_gts=self.peaks, tolerance=self.tolerance)
        
        if self.len < self.eps_len:
            rew, trun = 0, False
        else:
            rew, trun = rew, True
        term = False
        
        if self.render:
            self.act_buffer.append(act)
            self.obs_buffer.append(obs)
            self.preds_buffer.append(preds)

        if trun:
            return obs, rew, term, trun, {'preds': preds, 'TP': TP, 'FP': FP, 'FN': FN}
        else:
            return obs, rew, term, trun, {}
    

    def normalize(self, x):
        min, max = x.min(), x.max()
        scale, bias = (max - min)/2, (max + min)/2
        return (x-bias)/scale
    
    
    def _set_init_state(self, seed = None, status = 'train', idx = None):
        if status == 'train':
            random.seed(seed)
            data_file = random.choice(self.train_data_files) 
        elif status == 'test':
            data_file = self.test_data_files[idx]
        self.signal_pth = os.path.join('data', 'signal', data_file)
        peak_pth = os.path.join('data', 'peak', data_file)
        
        ecg = np.load(self.signal_pth).astype(np.float32)
        self.state = self.normalize(ecg)
        self.peaks = np.load(peak_pth).astype(np.float32)
    

    def plot_ecg(self):
        fig = plt.figure(figsize=(14, 3))
        plt.rcParams['lines.linewidth'] = 2
        gs = gridspec.GridSpec(nrows=2, ncols=2*self.eps_len + 1, 
                               width_ratios=[5] + [1, 5] * self.eps_len, 
                               height_ratios=[2.5, 1])

        len_filter, len_ecg = self.act_buffer[0].shape[0], self.obs_buffer[0].shape[1]
        t_filter = np.arange(len_filter)/self.fs
        t_ecg = np.arange(len_ecg)/self.fs
        diff = (len_ecg/5 - len_filter)/2
        
        ax = fig.add_subplot(gs[0:2, 0])
        ax.set_xlim(0, len_ecg/self.fs)
        ax.set_ylim(-1.2, 1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'$x_{1}$', fontsize=18)
        for peak in self.peaks:
            ax.axvline(x=peak/self.fs, color='r', linestyle='--')
        ax.plot(t_ecg, self.obs_buffer[0].reshape(-1), linewidth=2)
        
        for i in range(self.eps_len):
            filter, ecg, preds = self.act_buffer[i], self.obs_buffer[i+1].reshape(-1), self.preds_buffer[i]
            ecg[-1] = 0.0
            
            ax = fig.add_subplot(gs[0, 2 * i + 1])
            xlim = (-diff/self.fs, (len_filter+diff)/self.fs)
            ax.set_xlim(xlim)
            ax.set_ylim(-0.9, 1.2)
            ax.axis('off')
            ax.text((xlim[0]+xlim[1])/2, 0.9, f'$a_{i+1}$', ha='center', fontsize=18)
            ax.plot(t_filter, filter, 'darkorange', linewidth=2)
            
            ax = fig.add_subplot(gs[1, 2 * i + 1])
            ax.axis('off')
            ax.set_ylim(-0.7, 0.3)
            ax.text(0.5, 0.1, 'corr.', ha='center', fontsize=16)
            ax.annotate('', xy=(1.0, 0.5), # End of the arrow (pointing right)
                        xycoords='axes fraction', 
                        xytext=(0.0, 0.5), # Start of the arrow (horizontal center)
                        textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='black', linewidth=4))
            
            ax = fig.add_subplot(gs[0:2, 2 * i + 2])
            ax.set_xlim(0, len_ecg/self.fs)
            ax.set_ylim(-1.2, 1.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'$x_{i+2}$', fontsize=18)
            for peak in self.peaks:
                ax.axvline(x=peak/self.fs, color='r', linestyle='--')
            ax.plot(t_ecg, ecg, linewidth=2)
            ax.plot(preds/self.fs, ecg[preds], "x", color='magenta', markersize=10, markeredgewidth=2)

        return fig