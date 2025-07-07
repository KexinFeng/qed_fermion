import torch
import torch.nn as nn
import time

torch.manual_seed(0)

# Dummy model
class SimpleModel(nn.Module):
    def __init__(self, size=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size)
        )

    def forward(self, x):
        return self.net(x)

@torch.inference_mode()
def test():
    # Device setup
    device = torch.device("cuda")

    # Model and input
    model = SimpleModel().to(device)
    x = torch.randn(256, 1024, device=device)

    # Compile model with reduce-overhead mode
    compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=True)

    # Warm-up
    for _ in range(5):
        _ = compiled_model(x)

    # Measure time with compiled model (this should use CUDA Graph internally)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x = torch.randn(256, 1024, device=device)
        y = compiled_model(x)
        # print(torch.norm(y).item())
    torch.cuda.synchronize()
    end = time.time()

    print(f"Compiled (reduce-overhead) model time: {end - start:.4f} sec")

    # Reset and measure eager time
    model_eager = SimpleModel().to(device)
    for _ in range(5):  # also warm-up eager
        _ = model_eager(x)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x = torch.randn(256, 1024, device=device)
        y = model_eager(x)
        # print(torch.norm(y).item())

    torch.cuda.synchronize()
    end = time.time()

    print(f"Eager model time: {end - start:.4f} sec")

if __name__ == "__main__":
    test()