import torch

print("PyTorch:", torch.__version__)

device = "cpu"
print("Device:", device)

a = torch.arange(1024, device=device, dtype=torch.float32)
b = a * 2 + 1

print("Result:", b[:10].tolist())
print("CPU operation succeeded!")
