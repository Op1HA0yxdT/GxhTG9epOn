import torch
from torchprofile import profile_macs
try: 
    from models.archs.RetinexFormer_arch import RetinexFormer
    from models.archs.CSE_RetinexFormer_arch import CSE_RetinexFormer
except:
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    from basicsr.models.archs.CSE_RetinexFormer_arch import CSE_RetinexFormer


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# model = RetinexFormer(in_channels=3, out_channels=3, n_feat=24, num_blocks=[1,2,2],stage=1).to(device)
model = CSE_RetinexFormer(in_channels=3, out_channels=3, n_feat=24, num_blocks=[2,2,2],stage=1).to(device)
# model = RetinexFormer(stage=10).to(device) # TCA

input_tensor = torch.randn(1, 3, 256, 256).to(device) 

macs = profile_macs(model, input_tensor)
# macs = 0
flops = macs * 2
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

tflops = flops / (1024*1024*1024)

print(f"Model FLOPs (G): {tflops} G")
print(f"Model FLOPs (M): {tflops*1024} M")

print(f"Model MACs (G): {macs / (1024*1024*1024)} G")

print(f"Model params (M): {num_params / 1e6}")
print(f"Model params: {num_params}")

print(input_tensor.shape)
output = model(input_tensor)
print(output.shape)