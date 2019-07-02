import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import logging
log_file_name = "ppo_pong.log"
logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s', level = logging.DEBUG)

import pong_utils
device   = pong_utils.device

from common.multiprocessing_env import SubprocVecEnv
from pong_utils import preprocess_single, preprocess_batch

num_envs = 16
env_name = "Pong-v0"
#Hyper params:
hidden_size      = 32
lr               = 1e-3
num_steps        = 128
mini_batch_size  = 256
ppo_epochs       = 3
threshold_reward = 16
load_weight_n = 305000

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=4,
                               kernel_size=2,
                               stride=2,
                               )
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
#The second convolution layer takes a 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=2,
                               stride=2,
                               )
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
#The third convolution layer takes a 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=2,
                               stride=2,
                               )
        nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
#A fully connected layer takes the flattened frame from thrid convolution layer, and outputs 512 features
        self.conv4 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=2,
                               stride=2,
                               )
        nn.init.orthogonal_(self.conv4.weight, np.sqrt(2))
        self.lin = nn.Linear(in_features=5 * 5 * 32,
                             out_features=128)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))
#A fully connected layer to get logits for ππ
        self.pi = nn.Linear(in_features=128,
                                   out_features=6)
        nn.init.orthogonal_(self.pi.weight, np.sqrt(0.01))  # softmax 없어도 괜찮을까? -> relu 이기 때문에 괜찮다. 음수 안들어간다
#A fully connected layer to get value function
        self.value = nn.Linear(in_features=128,
                               out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)
    
    def forward(self, obs):

        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.reshape((-1, 5 * 5 * 32))

        h = F.relu(self.lin(h))
        pi = self.pi(h)
        value = F.relu(self.value(h))

        return pi, value


def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    f1 = env.reset()
    f2,_,_,_ = env.step(0)
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = preprocess_batch([f1,f2])
        prob, _ = model(state)
        dist = Categorical(logits = prob)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
        # print(total_reward)
    return total_reward

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage): # 전체 배치에서 mini_batch 를 만드는 것이다.
    batch_size = states.size(0)
    ids = np.random.permutation(batch_size)
    ids = np.split(ids, batch_size // mini_batch_size)
    for i in range(len(ids)):
        yield states[ids[i], :], actions[ids[i]], log_probs[ids[i]], returns[ids[i], :], advantage[ids[i], :]
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2): # training
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            with torch.autograd.detect_anomaly():
                pi, value = model(state)
                dist = Categorical(logits = pi)
                #     new_log_probs a= dist.log_prob(action)
                
                pi_a = pi.gather(1,action.unsqueeze(-1))
                # logging.warning(f'{pi_a} : pi_a')
                new_log_probs = torch.log(pi_a)

                ratio = (new_log_probs - old_log_probs).exp()
                # print("ratio :",  ratio)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage


                actor_loss  = - torch.min(surr1, surr2).mean()
                # print("actor loss", actor_loss)
                critic_loss = (return_.detach() - value).pow(2).mean()
                # print("critic loss", critic_loss)
                entropy = dist.entropy()

                loss = 0.5 * critic_loss + actor_loss  - 0.01 * entropy

                optimizer.zero_grad()
                
                loss.sum().backward()
                
                optimizer.step()
        print(loss.sum())

num_inputs  = envs.observation_space.shape
num_outputs = envs.action_space.n




model = ActorCritic().to(device) #return dist, v
# model.load_state_dict(torch.load(f'weight/pong_{load_weight_n}.pt'))
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 150000
frame_idx  = 0
test_rewards = []
early_stop = False

f1 = envs.reset()
f2 = envs.step([0]*num_envs)

from pong_utils import collect_trajectories
while not early_stop:
    log_probs , states, actions, rewards, next_state, masks, values = collect_trajectories(envs,model,num_steps)

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
            
    #경험 1세트 모은거로 학습하기
    #num_step 만큼 진행했을 때 reward 얼마였는지 구하기
#     next_state = preprocess_batch(next_state)
#     print("next_state shape: ", next_state.shape) # [16, 1, 80,80]
    _, next_value = model(next_state)
#     print("next_vlaue shape: " , next_value.shape)
    returns = compute_gae(next_value, rewards, masks, values)
    logging.debug(f"returns {returns} and shape is {len(returns)}, {len(returns[0])}" )
    returns = torch.cat(returns).detach()
#     logging.debug("after")
#     logging.debug(f"{returns} and shape is {returns.shape}" )
#     print(returns.shape, "what's happening here?")
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    logging.debug(f"log_probs {log_probs}")#" and shape is {len(log_probs)}, {len(log_probs[0])}" )
    logging.debug(f"values= {values}")#" and shape is {len(values)}, {len(values[0])}" )
    logging.debug(f"states= {states}")#" and shape is {len(states)}, {len(states[0])}" )
    logging.debug(f"actions= {actions}")#" and shape is {len(actions)}, {len(actions[0])}" )

    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)