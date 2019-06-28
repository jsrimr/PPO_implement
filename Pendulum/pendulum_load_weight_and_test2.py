import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import gym
# from ppo_practice import ActorCritic , test_env
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

env_name = "Pendulum-v0"
env = gym.make(env_name)
env = env.unwrapped
print("env made")
num_inputs  = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
hidden_size = 256

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value

def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
        print(total_reward)
    return total_reward

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device) #return dist, v
model.load_state_dict(torch.load('/home/jeffrey/RL_study/RL-Adventure-2/weight2/60000.pt'))
print("loaded weight")
model.eval()

test_env(vis=True)    
