import gymnasium
import torch
import os
import numpy as np
from datetime import datetime
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.sac.network import CNNActor, CNNQNetwork
from agents.sac.buffer import ReplayBuffer

class SAC():
    def __init__(self, env: gymnasium.Env, **kwargs) ->  None:
        self.env = env
        self._init_hyperparameters(**kwargs)
        self._seed()
        self._init_networks()
        self._init_buffer()
        # Logging
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/SAC_'+self.env_name+self.time_str)
        self.num_eps = 0
        self.eps_rets, self.eps_lens = [], []
        self.best_val_f1 = 0


    def learn(self):
        eps_ret, eps_len = 0, 0
        obs, info = self.env.reset(seed=self.num_eps+self.seed)
        for step in tqdm(range(self.total_steps)):
            # get action
            act = self.get_act(obs, step)
            assert self.env.action_space.contains(act), f"Invalid action {act} in the action space"

            # interact with the environment
            obs_next, rew, term, trun, info = self.env.step(act)
            done = term or trun

            # record transition in the replay buffer
            self.rb.add(obs, obs_next, act, rew, done)
            eps_ret, eps_len = eps_ret + rew, eps_len + 1

            # IMPORTANT: do not over look
            obs = obs_next

            if done:
                self.num_eps += 1
                self.eps_rets.append(eps_ret)
                self.eps_lens.append(eps_len)
                if self.use_tb:
                    self.writer.add_scalar('charts/episode_return', eps_ret, step)
                    self.writer.add_scalar('charts/episode_length', eps_len, step)
                
                # reset environment
                eps_ret, eps_len = 0, 0
                obs, info = self.env.reset(seed=self.num_eps+self.seed)

                if (self.num_eps)%self.save_freq == 0:
                    self.save_ckpt(ckpt_name=f'eps_{self.num_eps}.pt')
                if (self.num_eps-1)%self.val_freq == 0:
                    self.validate(seed=self.num_eps+self.seed, step=step)
        
            if step >= self.learning_starts:
                self.update(step)


    def update(self, step: int):
        data = self.rb.sample(self.batch_size)

        # Q-NETWORK UPDATE
        # compute target for the Q functions
        with torch.no_grad():
            act_next, log_prob_next, _ = self.actor.get_action(data.obs_next)
            qf1_next = self.qf1_target(data.obs_next, act_next)
            qf2_next = self.qf2_target(data.obs_next, act_next)
            min_qf_next = torch.min(qf1_next, qf2_next) - self.alpha*log_prob_next
            y = data.rew.flatten() + (1 - data.done.flatten()) * self.gamma * (min_qf_next).view(-1)
        # compute loss for the  Q functions
        qf_1 = self.qf1(data.obs, data.act).view(-1)
        qf_2 = self.qf2(data.obs, data.act).view(-1)
        qf1_loss = F.mse_loss(qf_1, y)
        qf2_loss = F.mse_loss(qf_2, y)
        qf_loss = qf1_loss + qf2_loss
        # update the Q functions
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        if self.use_tb:
            self.writer.add_scalar('charts/loss_q', qf_loss.item(), step)

        # POLICY UPDATE
        # update the policy
        if step % self.policy_freq == 0: # TD3 Delayed update support
            for _ in range(self.policy_freq):
                acts, log_prob, _ = self.actor.get_action(data.obs)
                qf1 = self.qf1(data.obs, acts)
                qf2 = self.qf2(data.obs, acts)
                min_qf = torch.min(qf1, qf2).view(-1)
                # negative sign for maximization
                actor_loss = -(min_qf - self.alpha * log_prob).mean()
                # update parameters
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            if self.use_tb:
                self.writer.add_scalar('charts/loss_policy', -actor_loss.item(), step)

        # UPDATE TARGET Q-NETWORKS
        if step % self.target_q_freq == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def get_act(self, obs: np.ndarray, step: int) -> np.ndarray:
        if self.ckpt_path == None and step<self.learning_starts:
            act = self.env.action_space.sample()
        else:
            act, _, _ = self.actor.get_action(torch.Tensor(obs).unsqueeze(0).to(self.device))
            act = act.squeeze(0).detach().cpu().numpy()
        return act
    

    def validate(self, seed=0, step=0):
        avg_eps_ret, avg_eps_len = 0, 0
        for idx in range(self.env.num_test):
            obs, _ = self.env.reset(seed, status = 'test', idx = idx)
            done = False
            eps_ret, eps_len = 0, 0
            while not done:
                _, _, act = self.actor.get_action(torch.Tensor(obs).unsqueeze(0).to(self.device))
                act = act.squeeze(0).detach().cpu().numpy()
                obs, rew, term, trun, _ = self.env.step(act, status = 'test')
                done = term or trun
                eps_ret += rew
                eps_len += 1
            avg_eps_len += eps_len
            avg_eps_ret += eps_ret
        avg_eps_len = avg_eps_len / self.env.num_test
        avg_eps_ret = avg_eps_ret / self.env.num_test
        print(f"Validation Episode Return: {avg_eps_ret}, Length: {avg_eps_len}")
        if self.use_tb:
            self.writer.add_scalar('validation/avg_episode_length', avg_eps_len, step)
            self.writer.add_scalar('validation/avg_episode_return', avg_eps_ret, step)
        if self.save_best and avg_eps_ret > self.best_val_f1:
            self.best_val_f1 = avg_eps_ret
            print(f'best with validation F-1 = {avg_eps_ret} saved')
            self.save_ckpt(ckpt_name=f'SAC_best.pt')


    def _init_hyperparameters(self, **kwargs):
        self.env_name = self.env.__class__.__name__
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {self.device}')
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
        print(f'Current time: {self.time_str}')

        self.seed = kwargs.get('seed', 0)
        self.use_tb = kwargs.get('use_tb', True)
        self.ckpt_path = kwargs.get('ckpt_path', None)
        
        self.total_steps = kwargs.get('total_steps', 100000)   # maximum number of iterations
        self.buffer_size = kwargs.get('buffer_size', 200000)   # replay buffer size
        self.batch_size = kwargs.get('batch_size', 512)      # batch size for updating network
        self.learning_starts = kwargs.get('learning_starts', self.batch_size) # start learning

        self.q_lr = kwargs.get('q_lr', 1e-4)                # learning rate for Q network
        self.policy_lr = kwargs.get('policy_lr', 1e-4)      # learning rate for policy network
        self.tau = kwargs.get('tau', 0.005)                 # for updating Q target
        self.gamma = kwargs.get('gamma', 0.99)              # forgetting factor
        self.alpha = kwargs.get('alpha', 0.2)               # entropy tuning parameter
        self.policy_freq = kwargs.get('policy_freq', 2)    # frequency for updating policy network
        self.target_q_freq = kwargs.get('target_q_freq', 1) # frequency for updating target network                    # displaying logs
        self.save_freq = kwargs.get('save_freq', 10000)     # frequency for saving the networks
        self.val_freq = kwargs.get('val_freq', 1000)       # frequency for validation
        self.render_val = kwargs.get('render_val', True)     # render validation
        self.save_best = kwargs.get('save_best', False)


    def _seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)


    def _init_networks(self):
        self.actor = CNNActor(self.env).to(self.device)
        self.qf1 = CNNQNetwork(self.env).to(self.device)
        self.qf2 = CNNQNetwork(self.env).to(self.device)
        self.qf1_target = CNNQNetwork(self.env).to(self.device)
        self.qf2_target = CNNQNetwork(self.env).to(self.device)
        if self.ckpt_path == None:
            print('Training from scratch')
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
        else:
            print('Training from the checkpoint in {self.ckpt_path}')
            self._load_ckpt(torch.load(self.ckpt_path, weights_only=True))
        self.q_optimizer = Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr)
        self.actor_optimizer = Adam(list(self.actor.parameters()), lr=self.policy_lr)


    def _init_buffer(self):
        self.rb = ReplayBuffer(
            self.env,
            self.buffer_size,
            self.device,
        )
 

    def save_ckpt(self, ckpt_name: str):
        directory = os.path.join('saved', f'sac_{self.env_name}_{self.time_str}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, ckpt_name)
        
        torch.save({"actor_state_dict": self.actor.state_dict(), 
            "qf1_state_dict": self.qf1.state_dict(),
            "qf2_state_dict": self.qf2.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict()
            }, path)

        print(f"Checkpoint saved to {path}")


    def _load_ckpt(self, ckpt: dict):
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.qf1.load_state_dict(ckpt["qf1_state_dict"])
        self.qf1_target.load_state_dict(ckpt["qf1_target_state_dict"])
        self.qf2.load_state_dict(ckpt["qf2_state_dict"])
        self.qf2_target.load_state_dict(ckpt["qf2_target_state_dict"])