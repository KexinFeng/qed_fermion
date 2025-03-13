import torch

# Setup: batch_size=5, input_dim=10
batch_size = 5
x = torch.randn(batch_size, 10, requires_grad=True)
weights = torch.randn(10, 1)
targets = torch.randn(batch_size, 1)

# Method 2: Manual batch gradient accumulation
def method2(x):
    gradients = []
    for i in range(batch_size):
        xi = x[i].unsqueeze(0)
        output = xi @ weights
        loss = (output - targets[i]).pow(2)
        grad_i = torch.autograd.grad(loss, xi, create_graph=True)[0]
        gradients.append(grad_i)
    return torch.cat(gradients).mean(dim=0)  # Average gradients

# Method 3: Using grad_outputs for batch processing
def method3(x):
    outputs = x @ weights
    losses = (outputs - targets).pow(2)  # Vector loss [5]
    return torch.autograd.grad(
        losses, 
        x, 
        grad_outputs=torch.ones_like(losses),
        create_graph=True
    )[0]

# Verification
grad2 = method2(x)
grad3 = method3(x)

print(torch.allclose(grad3, grad2, atol=1e-6))  # True (methods 1 & 2 equivalent)

