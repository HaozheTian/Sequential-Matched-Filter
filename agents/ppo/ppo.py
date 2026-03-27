import gymnasium
import torch
import os
import numpy as np
from datetime import datetime
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.ppo.network import Agent


class PPO():
    def __init__(self, env:gymnasium.Env, **kwargs):
        self.env = env
        self._init_hyperparameters(**kwargs)
        self._seed()
        self._init_networks()
        # Logging
        if self.use_tb:
            self.writer = SummaryWriter(log_dir='runs/PPO_'+self.env_name+self.time_str)
        self.global_step = 0
        self.num_eps = 0
        self.eps_rets, self.eps_lens = [], []
        self.best_val_f1 = 0
    
    
    
    def sample(self):
        obs_b = torch.zeros((self.batch_size,) + self.env.observation_space.shape, dtype=torch.float32, device=self.device)
        act_b = torch.zeros((self.batch_size,) + self.env.action_space.shape, dtype=torch.float32, device=self.device)
        logp_b = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        rew_b = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        done_b = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        v_b = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)
        adv_b = torch.zeros((self.batch_size,), dtype=torch.float32, device=self.device)

        # start collecting
        obs, _ = self.env.reset(seed=self.num_eps+self.seed)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        eps_ret, eps_len = 0, 0
        for step in range(self.batch_size):
            self.global_step += 1
            
            # interact with environment
            with torch.no_grad():
                act, logp, _, val, _ = self.agent.get_action_and_value(obs.unsqueeze(0))
            act, logp, val = act.squeeze(0), logp.squeeze(0), val.squeeze(0)
            obs_next, rew, term, trun, _ = self.env.step(act.cpu().numpy())
            obs_next = torch.tensor(obs_next, dtype=torch.float32, device=self.device)
            rew = torch.tensor([rew], dtype=torch.float32, device=self.device)
            done = torch.tensor([float(term or trun)], dtype=torch.float32, device=self.device)
            
            # record interaction
            obs_b[step] = obs
            act_b[step] = act
            logp_b[step] = logp
            rew_b[step] = rew
            done_b[step] = done
            v_b[step] = val.flatten()

            # IMPORTANT, EASY TO OVERLOOK
            obs = obs_next
            eps_ret, eps_len = eps_ret + rew.item(), eps_len + 1
            
            # reset if done
            if done:
                self.num_eps += 1
                self.eps_rets.append(eps_ret)
                self.eps_lens.append(eps_len)
                if self.use_tb:
                    self.writer.add_scalar('charts/episode_return', eps_ret, self.num_eps)
                    self.writer.add_scalar('charts/episode_length', eps_len, self.num_eps)

                # reset environment
                obs, _ = self.env.reset(seed=self.num_eps+self.seed)
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                eps_ret, eps_len = 0, 0
                
                if (self.num_eps)%self.save_freq == 0:
                    self.save_ckpt(ckpt_name=f'eps_{self.num_eps}.pt')
                if (self.num_eps - 1)%self.val_freq == 0:
                    self.validate(seed=self.num_eps+self.seed, step=self.global_step - 1)

        # compute GAE
        last_gaelam = 0
        for t in reversed(range(self.batch_size)):
            if t == self.batch_size - 1:
                with torch.no_grad():
                    next_value = self.agent.get_value(obs_next.unsqueeze(0)).squeeze(0)[0]
            else:
                next_value = v_b[t + 1]

            delta = rew_b[t] + (1 - done_b[t]) * self.gamma * next_value - v_b[t]
            adv_b[t] = last_gaelam = delta + (1 - done_b[t]) * self.gamma * self.gae_lambda * last_gaelam
        ret_b = adv_b + v_b
        return obs_b, act_b, logp_b, v_b, adv_b, ret_b
        


    def learn(self):
        for iteration in tqdm(range(self.num_iterations)):
            obs_b, act_b, logp_b, v_b, adv_b, ret_b = self.sample()
            
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lr_now = frac * self.lr
                self.optimizer.param_groups[0]["lr"] = lr_now
            
            b_inds = np.arange(self.batch_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    
                    _, logp_b_new, _, v_b_new, _ = self.agent.get_action_and_value(obs_b[mb_inds], act_b[mb_inds])
                    logp_b_new, v_b_new = logp_b_new.view(-1), v_b_new.view(-1)
                    
                    logratios = logp_b_new - logp_b[mb_inds]
                    ratios = logratios.exp()
                    mb_adv_b = adv_b[mb_inds]
                    if self.norm_adv:
                        mb_adv_b = (mb_adv_b - mb_adv_b.mean()) / (mb_adv_b.std() + 1e-8)
                    
                    # Policy loss
                    pg_loss1 = -mb_adv_b * ratios
                    pg_loss2 = -mb_adv_b * torch.clamp(ratios, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    # Value loss
                    v_loss = (v_b_new - ret_b[mb_inds]) ** 2
                    if self.clip_vloss:
                        v_clipped = v_b[mb_inds] + torch.clamp(
                            v_b_new - v_b[mb_inds],
                            -self.clip_coef,
                            self.clip_coef
                        )
                        v_loss_clipped = (v_clipped - ret_b[mb_inds]) ** 2
                        v_loss = torch.max(v_loss, v_loss_clipped)
                    v_loss = 0.5 * v_loss.mean()
                    
                    loss = pg_loss + self.vf_coef * v_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    


    def validate(self, seed=0, step=0):
        avg_eps_ret, avg_eps_len = 0, 0
        for idx in range(self.env.num_test):
            obs, _ = self.env.reset(seed, status = 'test', idx = idx)
            done = False
            eps_ret, eps_len = 0, 0
            while not done:
                _, _, _, _, act = self.agent.get_action_and_value(torch.Tensor(obs).unsqueeze(0).to(self.device))
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
            self.save_ckpt(ckpt_name=f'PPO_best.pt')


    def _init_hyperparameters(self, **kwargs):
        self.env_name = self.env.__class__.__name__
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {self.device}')
        self.time_str = datetime.now().strftime("_%m_%d_%Y_%H_%M")
        print(f'Current time: {self.time_str}')

        self.seed = kwargs.get('seed', 0)
        self.use_tb = kwargs.get('use_tb', True)
        self.ckpt_path = kwargs.get('ckpt_path', None)

        self.total_steps = kwargs.get('total_steps', 100000)
        self.batch_size = kwargs.get('batch_size', 500)
        self.num_iterations = self.total_steps // self.batch_size
        self.update_epochs = kwargs.get('update_epochs', 4)
        self.num_minibatches = kwargs.get('num_minibatches', 4)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        
        self.lr = kwargs.get('lr', 1e-4)
        self.anneal_lr = kwargs.get('anneal_lr', True)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gamma', 0.95)
        self.norm_adv = kwargs.get('norm_adv', True)
        self.clip_coef = kwargs.get('clip_coef', 0.2)
        self.clip_vloss = kwargs.get('clip_vloss', True)
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 0.5)
        self.render_val = kwargs.get('render_val', True)
        self.save_freq = kwargs.get('save_freq', 10000)
        self.val_freq = kwargs.get('val_freq', 1000)
        self.save_best = kwargs.get('save_best', False)



    def _seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.env.action_space.seed(self.seed)
        self.env.observation_space.seed(self.seed)



    def _init_networks(self):
        self.agent = Agent(self.env).to(self.device)
        if self.ckpt_path == None:
            print('Training from scratch')
        else:
            print('Training from the checkpoint in {self.ckpt_path}')
            self._load_ckpt(torch.load(self.ckpt_path, weights_only=True))
        self.optimizer = Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)



    def save_ckpt(self, ckpt_name: str):
        directory = os.path.join('saved', f'ppo_{self.env_name}_{self.time_str}')
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, ckpt_name)
        
        torch.save({"agent_state_dict": self.agent.state_dict()}, path)
        print(f"Checkpoint saved to {path}")



    def _load_ckpt(self, ckpt: dict):
        self.agent.load_state_dict(ckpt["agent_state_dict"])