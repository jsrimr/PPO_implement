import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.distributions import Normal
from torch.distributions import Categorical
import matplotlib.pyplot as plt

import logging
log_file_name = "ppo_pong.log"
logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s')


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from common.multiprocessing_env import SubprocVecEnv
from pong_util import preprocess_single, preprocess_batch

num_envs = 16
env_name = "Pong-v0"
#Hyper params:
hidden_size      = 32
lr               = 1e-3
num_steps        = 20
mini_batch_size  = 256
ppo_epochs       = 3
threshold_reward = 16
load_weight_n = 12000

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env

    return _thunk

envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

env = gym.make(env_name)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.constant_(m.bias, 0.1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNBase(nn.Module):
    def __init__(self, obs_shape, num_outputs,  hidden_size=512):
        super().__init__()

        # init_ = lambda m: init(m,
        #     nn.init.orthogonal_,
        #     lambda x: nn.init.constant_(x, 0),
        #     nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(32*6*6, hidden_size),
            nn.LeakyReLU()
        )

        # init_ = lambda m: init(m,
        #     nn.init.orthogonal_,
        #     lambda x: nn.init.constant_(x, 0))

        self.critic_linear = nn.Linear(hidden_size, num_outputs)

        self.train() # eval 의 반대모드

    def forward(self, inputs, rnn_hxs, ):
        x = self.main(inputs / 255.0)
    
        return self.critic_linear(x), x#return q_value, feature

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.base = CNNBase(mini_batch_size, num_outputs)
        self.actor = nn.Sequential(
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_outputs),
            nn.Softmax(dim = -1)
        )
        
    def forward(self, x):
        q_value, actor_features = self.base(x, num_outputs)
        
        policy    = self.actor(actor_features)
        
        return policy, q_value

def plot(frame_idx, rewards):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
    
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = preprocess_single(state)
        state = state.expand(1,1,80,80)
        prob, _ = model(state)
        dist = Categorical(prob)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
        print(total_reward)
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
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        
        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2): # training
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            pi, value = model(state)
            
            
            pi_a = pi.gather(1,action.unsqueeze(-1))
            logging.warning(f'{pi_a} : pi_a')
            new_log_probs = torch.log(pi_a)

            ratio = (new_log_probs - old_log_probs).exp()
            # print("ratio :",  ratio)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            with torch.autograd.detect_anomaly():
                actor_loss  = - torch.min(surr1, surr2).mean()
                # print("actor loss", actor_loss)
                critic_loss = (return_.detach() - value).pow(2).mean()
                # print("critic loss", critic_loss)
                # entropy = Categorical(pi).entropy()

                loss = 0.5 * critic_loss + actor_loss # - 0.001 * entropy

                optimizer.zero_grad()
                
                loss.sum().backward()
                
                optimizer.step()
        # print(loss.sum())

num_inputs  = envs.observation_space.shape
num_outputs = envs.action_space.n



state = envs.reset()

model = ActorCritic(state.shape, num_outputs, hidden_size).to(device) #return dist, v
model.load_state_dict(torch.load(f'weight/pong_{load_weight_n}.pt'))
optimizer = optim.Adam(model.parameters(), lr=lr)

max_frames = 150000
frame_idx  = 0
test_rewards = []
early_stop = False


while not early_stop:
    log_probs = []
    values    = []
    states    = []
    actions   = []
    rewards   = []
    masks     = []
    next_states = []

    envs.reset()
    n = len(envs.ps)
    # start all parallel agents
    envs.step([1]*n)
    # perform nrand random steps
    for _ in range(5):
        fr1, re1, _, _ = envs.step(np.random.choice(range(6),n))
        fr2, re2, _, _ = envs.step([0]*n)

    for _ in range(num_steps): #경험 모으기 - gpu 쓰는구나 . 하지만 여전히 DataParallel 들어갈 여지는 없어보인다. 
        #-> 아.. state 가 벡터 1개가 아닐 것 같다.. 16개네. gpu 쓸만하다. DataParallel 도 가능할듯?
        
        batch_input = preprocess_batch([fr1,fr2])

        state = preprocess_batch(state)
        
        dist, value = model(state)
        
        m = Categorical(dist)
        
        action = m.sample()
        
        
        next_state, reward, done, _ = envs.step(action.cpu().numpy()) #done 한다고 끝내질 않네??
        
        # logging.warning(f'dist[action] : {dist[action]}')
        log_prob = torch.log(dist[action])
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        states.append(state)
        actions.append(action)
        
        next_states.append(preprocess_batch(next_state))
        
        state = next_state
        frame_idx += 1
        
        
        if frame_idx % 1000 == 0 : # 1000번 마다 plot 그려주기
            print(frame_idx)
            torch.save(model.state_dict(),'weight/pong_{}.pt'.format(frame_idx+load_weight_n))

            test_reward = np.mean([test_env() for _ in range(10)])
            test_rewards.append(test_reward)
            # plot(frame_idx, test_rewards)
            print("test_reward : ", np.mean(test_rewards))
            if test_reward > threshold_reward: early_stop = True

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if done.any():
            break
            
    #경험 1세트 모은거로 학습하기
    
    #num_step 만큼 진행했을 때 reward 얼마였는지 구하기
    next_state = preprocess_batch(next_state)
    _, next_value = model(next_state)
    returns = compute_gae(next_value, rewards, masks, values)

    returns   = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values    = torch.cat(values).detach()
    states    = torch.cat(states)
    actions   = torch.cat(actions)
    advantage = returns - values
    
    ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)