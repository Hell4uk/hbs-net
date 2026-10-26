import sys

sys.path.insert(0, "src")

import torch

from hbs.config import ModelConfig
from hbs.model.mamba import SelectiveSSM

torch.manual_seed(0)
ssm = SelectiveSSM(32, 16)
x = torch.randn(1, 8, 32)
y_seq = ssm(x)
state = None
ys = []
for t in range(8):
    y_t, state = ssm.step(x[:, t : t + 1, :], state)
    ys.append(y_t)
y_step = torch.cat(ys, dim=1)
print("nan seq:", torch.isnan(y_seq).any().item())
print("nan step:", torch.isnan(y_step).any().item())
print("diff:", (y_seq - y_step).abs().max().item())

for t in range(8):
    d = (y_seq[:, t] - y_step[:, t]).abs().max().item()
    print(f"t={t} diff={d:.6f}")
