import torch
import time

device = torch.device("cuda")

class GraphRunner:
    def __init__(self, model, input_template):
        self.model = model
        self.input = input_template.clone()
        self.static_input = torch.cuda.graphs.make_graphed_callables(self.model)
        self.graph = torch.cuda.CUDAGraph()

        # Warm-up before graph capture
        for _ in range(3):
            self.model(self.input)

        # Capture CUDA Graph explicitly
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.static_input(self.input)

    def run(self, input_tensor):
        # Update inputs explicitly
        self.input.copy_(input_tensor)
        self.graph.replay()
        return self.static_input(self.input)

# Example model
class ExampleModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc = torch.nn.Linear(size, size)

    def forward(self, x):
        return torch.relu(self.fc(x))

size = 1024
batch_size = 256
inner_loops = 100  # smaller loops captured into CUDA graph
outer_loops = 10   # number of CUDA graph replays

model = ExampleModel(size).to(device)
runner = GraphRunner(model, torch.randn(batch_size, size, device=device))

# Run using nested loops: Outer loop (Python), Inner loop (CUDA graph)
torch.cuda.synchronize()
start = time.time()

for outer_idx in range(outer_loops):
    for inner_idx in range(inner_loops):
        input_tensor = torch.randn(batch_size, size, device=device)
        output = runner.run(input_tensor)

torch.cuda.synchronize()
print(f"Nested CUDA Graph loops executed in {time.time() - start:.4f}s")
