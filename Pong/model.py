class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=4,
                               kernel_size=6,
                               stride=2,
                               bias = False
                               )
        nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
#The second convolution layer takes a 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=4,
                               out_channels=8,
                               kernel_size=6,
                               stride=4,
                               )
        nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
#The third convolution layer takes a 9x9 frame and produces a 7x7 frame
        
        self.lin = nn.Linear(in_features=9 * 9 * 8,
                             out_features=256)
        nn.init.orthogonal_(self.lin.weight, np.sqrt(2))
#A fully connected layer to get logits for ππ
        self.pi = nn.Linear(in_features=256,
                                   out_features=6)
        nn.init.orthogonal_(self.pi.weight, np.sqrt(0.01))  # softmax 없어도 괜찮을까? -> relu 이기 때문에 괜찮다. 음수 안들어간다
#A fully connected layer to get value function
        self.value = nn.Linear(in_features=256,
                               out_features=1)
        nn.init.orthogonal_(self.value.weight, 1)
    
    def forward(self, obs):

        h = F.relu(self.conv1(obs))
        h = F.relu(self.conv2(h))
      
        h = h.reshape((-1, 9 * 9 * 8))

        h = F.relu(self.lin(h))
        pi = F.softmax(self.pi(h), dim = 1)
        value = F.relu(self.value(h))

        return pi, value
