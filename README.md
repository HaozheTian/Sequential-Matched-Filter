# RL-Driven Sequential Matched Filtering for ECG R-peak Detection

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2508.21652-b31b1b.svg)](https://arxiv.org/abs/2508.21652)

<img src="files/FigSchematic.png" alt="SMF Schematic" width="700"/>

<sub><i>
SMF sequentially applies matched filters (MFs) designed by a signal-aware neural network policy, and identifies ECG R-peaks at the final step. The policy is trained end-to-end using Reinforcement Learning (RL).
</i></sub>

</div>

---

## Overview

This is the **official implementation** of the Sequential Matched Filter (SMF) algorithm, presented in:

> **Machine Intelligence on the Edge: Interpretable Cardiac Pattern Localisation Using Reinforcement Learning**  
> Haozhe Tian, Qiyu Rao, Nina Moutonnet, Pietro Ferraro, Danilo Mandic  
> *Machine Intelligence Research*
> [[arXiv]](https://arxiv.org/abs/2508.21652)

SMF is a lightweight, interpretable ECG R-peak detector that uses matched filters (MFs) to iteratively refine signals.

---

## Requirements

Tested on **Ubuntu 22.04** with Python 3.8+. Set up the conda environment by running:

```bash
conda env create -f environment.yml
conda activate SMF
```

---

## Usage

### Quick Comparison of Pan-Tompkins, MF, and SMF

```bash
python compare.py
```

### R-peak Detection with Pre-trained Models

A demonstration notebook [`test.ipynb`](test.ipynb) is provided for running inference with trained checkpoints. The core is a `validate` function:

```python
from agents.ppo import PPO
from env import SMF

# Initialise environment: 3 MF steps, template length 8
env = SMF(eps_len=3, template_len=8)

# Load pre-trained PPO agent
agent = PPO(env, ckpt_path='saved/eps_len_3/PPO_best.pt', use_tb=False)

# Run validation
ecg, sig, peaks, preds, TP, FP, FN = validate(env, agent)
```

### Training from Scratch

| Argument | Description |
|---|---|
| `eps_len` | Number of sequential MF steps |
| `template_len` | Length of each matched filter template |
| `ckpt_path` | Path to a saved model checkpoint |

```bash
python train.py --agent ppo --eps_len 3 --template_len 8
```

---

## Citation

If you find the code in this research useful, please cite:

```bibtex
@article{tian2025machine,
  title={Machine Intelligence on the Edge: Interpretable Cardiac Pattern Localisation Using Reinforcement Learning},
  author={Tian, Haozhe and Rao, Qiyu and Moutonnet, Nina and Ferraro, Pietro and Mandic, Danilo},
  journal={arXiv preprint arXiv:2508.21652},
  year={2025}
}
```

If you use the PhysioNet/CinC 2017 dataset included in this repository, please also cite:

```bibtex
@inproceedings{clifford2017af,
  title={AF Classification from a Short Single Lead ECG Recording: The PhysioNet/Computing in Cardiology Challenge 2017},
  author={Clifford, Gari D and Liu, Chengyu and Moody, Benjamin and Lehman, Li-wei H and Silva, Ikaro and Li, Qiao and Johnson, Alistair E and Mark, Roger G},
  booktitle={2017 Computing in Cardiology (CinC)},
  pages={1--4},
  year={2017},
  organization={IEEE}
}
```
