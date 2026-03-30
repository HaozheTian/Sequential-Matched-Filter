import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from baselines.Pan_Tompkins import ECGProcessor
from agents.sac import SAC
from agents.ppo import PPO
from env import SMF

TOLERANCE = 5
DATA_DIR   = 'data'
RESULT_DIR = 'results'


# ── helpers ────────────────────────────────────────────────────────────────────

def load_data_splits(split: float = 0.7):
    """Return sorted train / test filename lists from data/peak/."""
    peak_dir = os.path.join(DATA_DIR, 'peak')
    files = sorted(
        [e.name for e in os.scandir(peak_dir) if e.is_file()],
        key=lambda x: int(x.split('.')[0])
    )
    cut = int(len(files) * split)
    return files[:cut], files[cut:]


def load_signal_and_peaks(filename: str, normalise: bool = False):
    signal = np.load(os.path.join(DATA_DIR, 'signal', filename)).astype(np.float32)
    peaks  = np.load(os.path.join(DATA_DIR, 'peak',   filename)).astype(np.float32)
    if normalise:
        lo, hi = signal.min(), signal.max()
        scale  = (hi - lo) / 2
        bias   = (hi + lo) / 2
        signal = (signal - bias) / scale
    return signal, peaks


def compute_metrics(TP: int, FP: int, FN: int, num_gts: int, num_preds: int) -> dict:
    """Return normalised precision, recall, F1 (rates, not counts)."""
    tp = TP / num_gts   if num_gts   > 0 else 0.0
    fp = FP / num_preds if num_preds > 0 else 0.0
    fn = FN / num_gts   if num_gts   > 0 else 0.0
    denom = 2 * tp + fp + fn
    precision = tp / (tp + fp)  if (tp + fp)  > 0 else 0.0
    recall    = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
    f1        = 2 * tp / denom  if denom      > 0 else 0.0
    return {'precision': precision, 'recall': recall, 'F1': f1}


def save_results(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved → {path}")


# ── agent inference ─────────────────────────────────────────────────────────

def get_action(agent, obs: np.ndarray) -> np.ndarray:
    t = torch.Tensor(obs).unsqueeze(0).to(agent.device)
    if isinstance(agent, SAC):
        _, _, act = agent.actor.get_action(t)
    elif isinstance(agent, PPO):
        _, _, _, _, act = agent.agent.get_action_and_value(t)
    else:
        raise NotImplementedError(f"Unknown agent type: {type(agent)}")
    return act.squeeze(0).detach().cpu().numpy()


# ── evaluation routines ──────────────────────────────────────────────────────

def evaluate_agent(env, agent, test_files: list[str]) -> dict:
    """Run the agent over all test files and return metric dict."""
    TP = FP = FN = 0
    for idx in range(env.num_test):
        ecg, _ = load_signal_and_peaks(test_files[idx])
        obs, _ = env.reset(seed=0, status='test', idx=idx)
        done   = False
        while not done:
            act = get_action(agent, obs)
            obs, _, term, trun, info = env.step(act, status='test')
            if term or trun:
                TP += info['TP']
                FP += info['FP']
                FN += info['FN']
                break

    num_gts   = TP + FN
    num_preds = TP + FP
    return compute_metrics(TP, FP, FN, num_gts, num_preds)


def evaluate_pan_tompkins(test_files: list[str]) -> dict:
    """Run Pan-Tompkins baseline on all test files and return metric dict."""
    processor  = ECGProcessor()
    all_signal, all_peaks = [], []
    for idx, fname in enumerate(test_files):
        sig, peaks = load_signal_and_peaks(fname, normalise=True)
        all_signal.append(sig)
        all_peaks.append(peaks + idx * 250)

    signal   = np.concatenate(all_signal)
    peaks_gt = np.concatenate(all_peaks)

    preds, *_ = processor.find_qrs_peaks(signal)
    hits      = np.any(
        np.abs(preds.reshape(-1, 1) - peaks_gt.reshape(1, -1)) < TOLERANCE,
        axis=1
    )
    TP = int(hits.sum())
    FP = len(preds) - TP
    FN = len(peaks_gt) - TP
    return compute_metrics(TP, FP, FN, len(peaks_gt), len(preds))


def collect_plot_data(env, agent, test_files: list[str], idx: int = 0):
    """Run agent on a single test file and return arrays needed for plot_sig."""
    fname      = test_files[idx]
    ecg, peaks = load_signal_and_peaks(fname)
    obs, _     = env.reset(seed=0, status='test', idx=idx)
    done       = False
    while not done:
        act = get_action(agent, obs)
        obs, _, term, trun, info = env.step(act, status='test')
        if term or trun:
            sig   = env.state.copy()
            preds = info['preds'].astype(np.int32)
            break
    return ecg, sig, peaks.astype(np.int32), preds


# ── experiment table ─────────────────────────────────────────────────────────

EXPERIMENTS = [
    # (label, agent_cls, eps_len, template_len, ckpt_subdir)
    ('MF-PPO',  PPO,       1,        8,           'eps_len_1/PPO_best.pt'),
    ('MF-SAC',  SAC,       1,       12,           'eps_len_1/SAC_best.pt'),
    ('SMF-PPO', PPO,       3,        8,           'eps_len_3/PPO_best.pt'),
    ('SMF-SAC', SAC,       3,       12,           'eps_len_3/SAC_best.pt'),
]

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    _, test_files = load_data_splits()
    results = []

    # Baseline
    print("Evaluating Pan-Tompkins …")
    metrics = evaluate_pan_tompkins(test_files)
    results.append({'method': 'Pan-Tompkins', **{k: f"{v:.3f}" for k, v in metrics.items()}})
    print(f"  {metrics}")

    # RL agents
    for label, AgentCls, eps_len, tmpl_len, ckpt_rel in EXPERIMENTS:
        print(f"Evaluating {label} …")
        ckpt_path = os.path.join('saved', ckpt_rel)
        env       = SMF(render=False, eps_len=eps_len, template_len=tmpl_len)
        agent     = AgentCls(env, ckpt_path=ckpt_path, use_tb=False)
        metrics   = evaluate_agent(env, agent, test_files)
        results.append({'method': label, **{k: f"{v:.3f}" for k, v in metrics.items()}})
        print(f"  {metrics}")

    save_results(results, os.path.join(RESULT_DIR, 'comparison.csv'))


if __name__ == '__main__':
    main()