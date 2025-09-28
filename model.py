import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# ANFIS Model
# -------------------------------
class FuzzyLayer(nn.Module):
    def __init__(self, input_dim, num_mfs=2):
        super().__init__()
        self.centers = nn.Parameter(torch.rand(input_dim, num_mfs))
        self.sigmas  = nn.Parameter(torch.rand(input_dim, num_mfs))

    def forward(self, x):
        x_exp = x.unsqueeze(2)                      
        return torch.exp(- (x_exp - self.centers)**2
                         / (2 * self.sigmas**2 + 1e-6))  # [B, F, M]

class RuleLayer(nn.Module):
    def forward(self, x):
        # flatten all fuzzy memberships into rule strengths
        return x.view(x.size(0), -1)  # [B, F*M]

class ConsequentLayer(nn.Module):
    def __init__(self, num_rules, output_dim=3):
        super().__init__()
        self.consequents = nn.Parameter(torch.rand(num_rules, output_dim))

    def forward(self, x):  # x : Batch_size,num_rules and output would be Batch_size,output_dim
        norm = x / (x.sum(1, keepdim=True) + 1e-6)  
        return norm @ self.consequents              

class ANFIS(nn.Module):
    def __init__(self, input_dim, num_mfs=2, output_dim=3):
        super().__init__()
        self.fuzzy  = FuzzyLayer(input_dim, num_mfs)
        self.rule   = RuleLayer()
        self.conseq = ConsequentLayer(input_dim * num_mfs, output_dim)

    def forward(self, x):
        return self.conseq(self.rule(self.fuzzy(x)))