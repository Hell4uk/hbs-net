import sys

sys.path.insert(0, "src")

import torch

from hbs.model.mamba import SelectiveSSM

torch.manual_seed(0)
ssm = SelectiveSSM(32, 16)
x = torch.randn(1, 8, 32)
xz = ssm.in_proj(x)
x_in, z = xz.chunk(2, -1)
xi = x_in.transpose(1, 2)
xi_seq = ssm.conv1d(xi)[..., : xi.shape[-1]].transpose(1, 2)

state = None
convs = []
for t in range(8):
    xt = x_in[:, t]
    if state is None:
        cw = 3
        state = torch.zeros(1, 32, 16 + cw)
    conv_state = state[..., 16:]
    window = torch.cat([conv_state, xt.unsqueeze(-1)], dim=-1)
    xc = ssm.conv1d(window)[..., 3]
    convs.append(xc)
    state = torch.cat([state[..., :16], window[..., 1:]], dim=-1)
conv_step = torch.stack(convs, dim=1)
print("conv diff:", (xi_seq - conv_step).abs().max().item())
