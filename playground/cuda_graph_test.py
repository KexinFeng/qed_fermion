import torch

x = torch.ones(128, device='cuda')
y = torch.empty_like(x)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y.copy_(x + 1)
graph.replay()

dbstop = 1
