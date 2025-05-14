import torch

def try_full_inside_graph():
    x = torch.empty((32,), device='cuda')  # dummy to init CUDA

    graph = torch.cuda.CUDAGraph()
    x_static = torch.empty_like(x)

    with torch.cuda.graph(graph):
        # TRY to do unsafe allocation
        # y = torch.full((32,), 1.0, device='cuda')
        y = torch.zeros((32,), device='cuda')
        x_static.copy_(y)

    graph.replay()
    print("Replay worked, value:", x_static)

try_full_inside_graph()
