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
    x_static = torch.randn(2560, 1024, device=device)

    # Compile model with reduce-overhead mode
    compiled_model = torch.compile(model, mode='reduce-overhead', fullgraph=True, dynamic=False)

    # Warm-up
    for _ in range(5):
        _ = compiled_model(x_static)

    # Measure time with compiled model (this should use CUDA Graph internally)
    x = torch.randn(2560, 1024, device=device)
    torch.cuda.synchronize()
    start = time.time()
    x_static.copy_(x)
    for _ in range(100):
        y = compiled_model(x_static)
        # print(torch.norm(y).item())
    torch.cuda.synchronize()
    end = time.time()

    print(f"Compiled (reduce-overhead) model time: {end - start:.4f} sec")

    # CUDA Graph test
    model_graph = SimpleModel().to(device)
    static_x = torch.randn(2560, 1024, device=device)
    # static_y = torch.empty(2560, 1024, device=device)
    stream = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    # Warm-up
    for _ in range(5):
        _ = model_graph(static_x)
    torch.cuda.synchronize()
    # Capture
    with torch.cuda.graph(g, stream=stream):
        y = model_graph(static_x)
    torch.cuda.synchronize()

    x = torch.randn(2560, 1024, device=device)
    # Timing CUDA Graph replay
    torch.cuda.synchronize()
    start = time.time()
    static_x.copy_(x)
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"CUDA Graph replay time: {end - start:.4f} sec")

    # Reset and measure eager time
    model_eager = SimpleModel().to(device)
    x = torch.randn(2560, 1024, device=device)
    for _ in range(5):  # also warm-up eager
        _ = model_eager(x)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = model_eager(x)
        # print(torch.norm(y).item())

    torch.cuda.synchronize()
    end = time.time()

    print(f"Eager model time: {end - start:.4f} sec")

if __name__ == "__main__":
    test()

# Note
# torch                              2.1.2+cu121
# torchaudio                         2.1.2+cu121
# torchvision                        0.16.2+cu121
