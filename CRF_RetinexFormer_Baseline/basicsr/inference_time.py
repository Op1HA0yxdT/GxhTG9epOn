import torch, time
from torch import nn
from models.archs.RetinexFormer_arch import RetinexFormer
from models.archs.CSE_RetinexFormer_arch import CSE_RetinexFormer
# ---------------------------------------------------------------------
# import or define your two models here
# ---------------------------------------------------------------------

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# ---------------------------------------------------------------------
# helper: measure avg time (ms) over N runs after warm-up
# ---------------------------------------------------------------------
def benchmark(model: nn.Module,
              input_shape=(1, 3, 256, 256),
              warmup=10,
              runs=50):
    model.eval().to(device)
    dummy = torch.randn(*input_shape, device=device)
    
    # CUDA timer events for accurate GPU timing; fallback to time.perf_counter on CPU
    if device.type == "cuda:1":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        # warm-up
        for _ in range(warmup):
            _ = model(dummy)
        # timed runs
        total_ms = 0.0
        for _ in range(runs):
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            total_ms += starter.elapsed_time(ender)      # milliseconds
        return total_ms / runs
    else:
        # CPU fallback
        for _ in range(warmup):
            _ = model(dummy)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model(dummy)
        end = time.perf_counter()
        return (end - start) * 1000 / runs               # milliseconds


# ---------------------------------------------------------------------
# instantiate and benchmark the two models
# ---------------------------------------------------------------------
model_a = RetinexFormer(in_channels=3,
                        out_channels=3,
                        n_feat=40,
                        stage=1,
                        num_blocks=[1, 2, 2])

model_b = CSE_RetinexFormer(in_channels=3,
                            out_channels=3,
                            n_feat=24,
                            stage=1,
                            num_blocks=[1, 1, 2])

time_a = benchmark(model_a)
time_b = benchmark(model_b)

print(f"RetinexFormer       : {time_a:.2f} ms / image (256×256)")
print(f"CSE_RetinexFormer   : {time_b:.2f} ms / image (256×256)")
