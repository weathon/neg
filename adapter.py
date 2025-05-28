import torch.nn as nn
import torch 
import random

class Adapter(nn.Module):
    def __init__(self):
        super(Adapter, self).__init__()
        self.conv1 = nn.Conv2d(40, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        torch.nn.init.zeros_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        torch.nn.init.zeros_(self.conv2.bias)
        torch.nn.init.zeros_(self.conv3.bias)
            
    def forward(self, attention, t, cfg_scale):
        t = t / 1000
        cfg_scale = torch.tensor(cfg_scale, dtype=torch.float32).to(attention.device)/10
        
        t = t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        t = t.repeat(1, 1, attention.size(1), attention.size(2))
        
        cfg_scale = cfg_scale.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cfg_scale = cfg_scale.repeat(1, 1, attention.size(1), attention.size(2))
        
        attention = attention.unsqueeze(0)
        
        x = torch.cat((attention, t, cfg_scale), dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x
    