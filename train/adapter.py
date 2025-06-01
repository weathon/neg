import torch.nn as nn
import torch 
import random

class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        self.conv1 = nn.Conv2d(25, 32, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 3, kernel_size=1, stride=1, padding=0)
        
    
    def train_init(self):
        # init that it output 1, 9, -9
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.train()
        
        for epoch in range(5000):
            optimizer.zero_grad()
            attention = torch.randn(24, 128, 128)
            t = torch.randn(1) * 1000
            pred = self.forward(attention, t)
            loss = (pred.mean((0, 2, 3)) - torch.tensor([1, 5, -5])).abs().mean()
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")
            
             
    def forward(self, attention, t):
        t = t / 1000
        
        t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 1, attention.size(1), attention.size(2))
        
        attention = attention.unsqueeze(0)
        
        x = torch.cat((attention, t), dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x
    