import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
# from ppo_practice import ActorCritic , test_env

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

env_name = "Pong-v0"
env = gym.make(env_name)
print("env made")

hidden_size = 32

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNBase(nn.Module):
    def __init__(self, obs_shape, num_outputs,  hidden_size=512):
        super().__init__()



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


        self.critic_linear = nn.Linear(hidden_size, num_outputs)

        self.train() # eval 의 반대모드

    def forward(self, inputs, rnn_hxs, ):
        x = self.main(inputs / 255.0)
    
        return self.critic_linear(x), x#return q_value, feature

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.base = CNNBase(None, num_outputs)
        self.actor = nn.Sequential(
            nn.Linear(512, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_outputs),
            nn.Softmax(dim = -1)
        )
        
    def forward(self, x):
        num_outputs = 6
        q_value, actor_features = self.base(x, num_outputs)
        
        policy    = self.actor(actor_features)
        
        return policy, q_value


def preprocess_single(image, bkg_color = np.array([144, 72, 17])):
    img = np.mean(image[34:-16:2,::2]-bkg_color, axis=-1)/255.
    return torch.from_numpy(img).float().to(device)

def preprocess_batch(images, bkg_color = np.array([144, 72, 17])):
    list_of_images = np.asarray(images)
    if len(list_of_images.shape) < 5:
        list_of_images = np.expand_dims(list_of_images, 1)
    # subtract bkg and crop
    list_of_images_prepro = np.mean(list_of_images[:,:,34:-16:2,::2]-bkg_color,
                                    axis=-1)/255.
    # print(list_of_images_prepro.shape)
    # batch_input = np.swapaxes(list_of_images_prepro,0,1)
    # print(batch_input.shape)
    return torch.from_numpy(list_of_images_prepro).float().to(device)
            
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

if __name__ == "__main__":
    model = ActorCritic(None, 6, 32).to(device) #return dist, v
    model.load_state_dict(torch.load('weight/pong_1000.pt'))
    print("loaded weight")
    model.eval()

    test_env(vis=False)   
