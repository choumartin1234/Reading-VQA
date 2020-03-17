import torch
import torch.nn as nn

class MutanFusion(nn.Module):
    def __init__(self, dim_a, dim_ha, dim_b, dim_hb, dim_mm, dropout_a=0.5, dropout_b=0.5, R=5):
        super().__init__()
        self.R = R
        self.linear_a = nn.Linear(dim_a, dim_ha)
        self.dropout_a = nn.Dropout(dropout_a)
        self.linear_b = nn.Linear(dim_b, dim_hb)
        self.dropout_b = nn.Dropout(dropout_b)
        self.list_linear_ha = nn.ModuleList([nn.Linear(dim_ha, dim_mm) for _ in range(R)])
        self.list_linear_hb = nn.ModuleList([nn.Linear(dim_hb, dim_mm) for _ in range(R)])

    def forward(self, input_a, input_b):
        batch_size = input_a.size(0)
        R = self.R
        x_a = torch.tanh(self.linear_a(self.dropout_a(input_a)))
        x_b = torch.tanh(self.linear_b(self.dropout_b(input_b)))
        x_mm = []
        for i in range(R):
            x_ha = torch.tanh(self.list_linear_ha[i](x_a))
            x_hb = torch.tanh(self.list_linear_hb[i](x_b))
            x_mm.append(torch.mul(x_ha, x_hb))
        x_mm = torch.stack(x_mm, 1).sum(1)
        return x_mm
