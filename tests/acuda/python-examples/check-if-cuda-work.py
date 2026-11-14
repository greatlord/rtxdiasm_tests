import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))

    a = torch.arange(1024, device="cuda", dtype=torch.float32)
    b = a * 2 + 1
    torch.cuda.synchronize()

    print("Result:", b[:10].cpu().tolist())
    print("GPU operation succeeded!")
