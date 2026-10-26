import sys

sys.path.insert(0, "src")

import torch
import torch.nn.functional as F

from hbs.model.mamba import SelectiveSSM

torch.manual_seed(0)
ssm = SelectiveSSM(32, 16)
x = torch.randn(1, 8, 32)
xz = ssm.in_proj(x)
x_in, z = xz.chunk(2, -1)
xi = x_in.transpose(1, 2)
xi = ssm.conv1d(xi)[..., : xi.shape[-1]].transpose(1, 2)
xi_s = F.silu(xi)
B, C, dt = ssm.x_proj(xi_s).chunk(3, -1)
dt = ssm.dt_proj(dt)
dtf = F.softplus(dt)
A_bar = torch.exp(dtf.unsqueeze(-1) * ssm.A)
B_bar = dtf.unsqueeze(-1) * B.unsqueeze(-2)
inp = B_bar * xi_s.unsqueeze(-1)

h = ssm._parallel_scan(A_bar, inp)
h_seq = torch.zeros_like(h)
hh = torch.zeros(1, 32, 16)
for t in range(8):
    hh = A_bar[:, t] * hh + inp[:, t]
    h_seq[:, t] = hh
print("h diff:", (h - h_seq).abs().max().item())

state = None
ys = []
for t in range(8):
    y_t, state = ssm.step(x[:, t : t + 1, :], state)
    ys.append(y_t)
y_step = torch.cat(ys, dim=1)
y_seq = ssm(x)
print("full diff:", (y_seq - y_step).abs().max().item())
