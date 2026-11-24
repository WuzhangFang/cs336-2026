import torch
import torch.nn as nn
from cs336_basics.nn_utils import cross_entropy


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


device = "cuda"
batch_size, in_features, out_features = 16, 16, 16

model = ToyModel(in_features, out_features).to(device)  # parameters in FP32
x = torch.randn(batch_size, in_features, device=device)
targets = torch.randint(0, out_features, (batch_size,), device=device)

with torch.autocast(device_type=device, dtype=torch.float16):
    print("parameters:", {p.dtype for p in model.parameters()})
    fc1_out = model.fc1(x)
    print("fc1 output:", fc1_out.dtype)
    ln_out = model.ln(model.relu(fc1_out))
    print("ln output: ", ln_out.dtype)
    logits = model.fc2(ln_out)
    print("logits:    ", logits.dtype)
    loss = cross_entropy(logits, targets)
    print("loss:      ", loss.dtype)

loss.backward()
print("gradients: ", {p.grad.dtype for p in model.parameters()})
