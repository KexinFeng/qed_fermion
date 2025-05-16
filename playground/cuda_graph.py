import torch

def try_full_inside_graph():
    x = torch.empty((32,), device='cuda')  # dummy to init CUDA

    x_static = torch.empty_like(x)

    # Warm up
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # Warm up
        y1 = torch.full((32,), 1.0, device='cuda')
        y = torch.zeros((32,), device='cuda')
        x_static.copy_(y1)
        
        x = x_static + y
        x_static.copy_(y)

    s.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # TRY to do unsafe allocation
        y1 = torch.full((32,), 1.0, device='cuda')
        y = torch.ones((32,), device='cuda')
        x_static.copy_(y1)

        x = x_static + y
        x_static.copy_(y)
        dbstop = 1

    graph.replay()
    print("Replay worked!\nvalue:", x_static)

try_full_inside_graph()
