import tyro
from agents.ppo import PPO
from agents.sac import SAC
from env import SMF
from dataclasses import dataclass

from torch.utils import tensorboard
from typing import Literal

@dataclass
class Args:
    render: bool = False
    eps_len: int = 3
    template_len: int = 8
    agent: Literal["sac", "ppo"] = "ppo"
    seed: int = 0
    use_tb: bool = True
    save_best: bool = True


AGENTS = {"sac": SAC, "ppo": PPO}


def main(args: Args):
    env = SMF(render=args.render, eps_len=args.eps_len, template_len=args.template_len)
    agent_cls = AGENTS[args.agent]
    agent = agent_cls(env, use_tb=args.use_tb, seed=args.seed, save_best=args.save_best)
    agent.learn()


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)