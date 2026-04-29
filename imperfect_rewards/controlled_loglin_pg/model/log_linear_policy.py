import torch
import torch.nn as nn
import torch.nn.functional as F


class LogLinearPolicy(nn.Module):
    def __init__(self, input_dim: int, temperature: float = 1):
        super(LogLinearPolicy, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1, bias=False)
        self.temperature = temperature

    def forward(self, output_features: torch.Tensor):
        logits = self.linear(output_features).squeeze(dim=-1)
        logits /= self.temperature
        return F.softmax(logits, dim=1)
