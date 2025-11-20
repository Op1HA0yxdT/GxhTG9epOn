from thop import profile
import torch
import time
from net.CIDNet import CIDNet, CSE_CIDNet

device = 'cuda'

# model = CSE_CIDNet(channels=[24, 36, 80]).to(device)
model = CSE_CIDNet(channels=[32, 48, 64]).to(device)
input = torch.rand(1, 3, 400, 600).to(device)

model.eval()

# warm-up (not timed)
with torch.no_grad():
    for _ in range(10):
        _ = model(input)

# measure average inference time
n_runs = 20
torch.cuda.synchronize()
time_start = time.time()

with torch.no_grad():
    for _ in range(n_runs):
        _ = model(input)

torch.cuda.synchronize()
time_end = time.time()

avg_time = (time_end - time_start) / n_runs  # seconds per inference
print(f"Average inference time: {avg_time:.6f} s  ({avg_time*1000:.3f} ms)")

# parameter count
n_param = sum(p.nelement() for p in model.parameters())
print(f"n_paras: {n_param / 2**20:.3f} M")

# FLOPs (MACs) with thop
macs, params = profile(model, inputs=(input,))
print(f"FLOPs: {macs / 2**30:.3f} G")
