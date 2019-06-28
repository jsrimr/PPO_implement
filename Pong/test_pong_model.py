import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import gym
# from ppo_practice import ActorCritic , test_env

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

env_name = "Pong-v0"
env = gym.make(env_name)
print("env made")

hidden_size = 32
learning_rate = 0.0005

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class PPO(nn.Module):
    def __init__(self, input_size=2, hidden_size=512, output_size=6):
        super().__init__()
        
        self.base = nn.Sequential(
            #80 -> 40
            nn.Conv2d(input_size, 32, 4, stride=2),
            nn.LeakyReLU(),
            #40->20
            nn.Conv2d(32, 64, 4, stride=2),
            nn.LeakyReLU(),
            #20->20
            nn.Conv2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(32*16*16, hidden_size),
            nn.LeakyReLU()
        )      
        
        self.data = []

        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc_pi = nn.Linear(256, output_size)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=-1):
        x = self.base(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.base(x)
        x = F.leaky_relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def forward(self,x):
        return self.pi(x), self.v(x)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        data = np.array(self.data)
        s = data[:,0]
        a = data[:,1].astype("int32")
        r = data[:,2].astype("float32")
        s_prime = data[:,3]
        prob_a = data[:,4].astype("float32")      
        done = data[:,5].astype("int32")
        
        
        return torch.cat([*s]).cuda(), torch.from_numpy(a).cuda().long(), torch.from_numpy(r).cuda().float(), torch.cat([*s_prime]).cuda(), torch.from_numpy(prob_a).cuda().float(), torch.from_numpy(done).cuda().float()
       
    def train_net(self):
        s, a, r, s_prime, prob_a, done_mask = self.make_batch()

        for i in range(K_epoch):  # 배치 하나를 K_epoch번 반복함
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a.unsqueeze(-1))
            
            
            # a/b == exp(log(a)-log(b))
            #새로운 policy 가 특정 action 에 주는 확률과 이전 policy 가 특정 action 에 주던 확률을 비교한다
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.float()))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + \
                F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            print(loss.mean())
            self.optimizer.step()
        
            del loss, pi , advantage

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
    batch_input = np.swapaxes(list_of_images_prepro,0,1)
    return torch.from_numpy(batch_input).float().to(device)
            
def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:

        state2, _, _, _ = env.step(0)        
        state = preprocess_batch([state,state2])

        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
        print(total_reward)
    return total_reward

if __name__ == "__main__":
    model = PPO().to(device) #return dist, v
    model.load_state_dict(torch.load('weight/pong_1000.pt'))
    print("loaded weight")
    model.eval()

    test_env(vis=False)   
