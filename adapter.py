import torch.nn as nn
import torch 
import random

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(24, 32, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x
    

if __name__ == "__main__":
    # calculate the number of parameters in the model
    model = Policy()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params}")
    
    # toy policy gradient, output as close to 0 as possible
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(1000):
        x = torch.randn(1, 24, 64, 64)
        p_action = model(x)
        p_action = p_action if p_action.mean() > 0.5 else 1 - p_action
        log_p_action = torch.log(p_action)
        reward = 1 if torch.mean(p_action) > 0.5 else 0
        reward = torch.tensor(reward, dtype=torch.float32)
        loss = (-log_p_action * reward).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}, Reward: {reward.item()}")
        