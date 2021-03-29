import torch
from torch import nn
import numpy as np

class SiameseNet(nn.Module):
    def __init__(self, **kwargs):
        super(SiameseNet, self).__init__()
        self.s = nn.Sequential( 
            nn.Linear(kwargs["input_size"],1024), 
             nn.ReLU(),
             nn.Linear(1024, 1024), 
             nn.ReLU(),
             nn.Linear(1024, 512), 
             nn.ReLU(),
             nn.Linear(512,kwargs["output_size"]),
             nn.ReLU())
        

    def forward(self, x1, x2=None):
        if x2 == None:
            z1 = self.s(x1)
            return z1
        else:
            z1 = self.s(x1)
            z2 = self.s(x2)
            return z1, z2
    
class AE(nn.Module):
    def __init__(self, **kwargs):
        super(AE,self).__init__()
        self.encoder = nn.Sequential( 
            nn.Linear(kwargs["input_size"],1024), 
             nn.ReLU(),
             nn.Linear(1024,512), 
             nn.ReLU(),
             nn.Linear(512,kwargs["code_size"]))
            
        self.decoder = nn.Sequential( 
            nn.Linear(kwargs["code_size"],512), 
             nn.ReLU(),
             nn.Linear(512,1024), 
             nn.ReLU(),
             nn.Linear(1024,kwargs["input_size"]), 
             nn.Sigmoid())
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class SpectralNet(nn.Module):
    def __init__(self, **kwargs):
        super(SpectralNet, self).__init__()
        self.s = nn.Sequential( 
            nn.Linear(kwargs["input_size"],1024), 
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,kwargs["output_size"]),
            nn.Tanh())

    def make_ortho_weights(self,x):
        eps= 1e-7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_sym = torch.mm(x.t(), x) 
        x_sym += torch.eye(x_sym.shape[1],device=device)*eps
        L = torch.cholesky(x_sym)
        ortho_weights = ((x.shape[0])**(1/2) )* (torch.inverse(L)).t()
        return ortho_weights
    def ortho_update(self,x):
        ortho_weights = self.make_ortho_weights(x)
        self.W_ortho = ortho_weights
        
    def forward(self, x, ortho_step=False):
        x_net = self.s(x)
        if ortho_step:
            self.ortho_update(x_net)
        y = x_net@self.W_ortho          

        return y
    
