# GradFlow

A lightweight scalar autograd engine and neural network library built from scratch in pure Python with **zero external machine learning libraries**.

GradFlow implements reverse-mode automatic differentiation (backpropagation over a dynamically built DAG) and gradient descent from first principles.

---

## Features

- **Zero ML Dependencies**: Built with pure Python — no PyTorch, NumPy, or TensorFlow needed.
- **Scalar Autograd Engine**: Tracks computational history and computes exact gradients via topological sort.
- **Supported Operations**: Addition, subtraction, multiplication, division, powers, and ReLU activation.
- **PyTorch-like Neural Network API**: Modular components including `Neuron`, `Layer`, and `MLP` with `.parameters()` and `.zero_grad()`.
- **Graph Visualization**: Optional computational graph rendering via Graphviz.

---

## Quickstart

### 1. Autograd & Backpropagation

```python
from Value import Value

# Define scalar values
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

# Forward pass (computational graph)
d = a * b + c
L = d.relu()

# Backward pass (computes gradients)
L.backward()

print(f"Loss: {L.data}")     # 4.0
print(f"dL/da: {a.grad}")    # -3.0
print(f"dL/db: {b.grad}")    # 2.0
```

### 2. Training a Neural Network with Gradient Descent

```python
from Value import Value
from neural_network import MLP

# 2-input, two hidden layers of 4 neurons, 1 output
model = MLP(2, [4, 4, 1])

# Tiny toy dataset
xs = [
    [2.0, 3.0],
    [3.0, -1.0],
    [0.5, 1.0],
    [1.0, 1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# Training loop (Gradient Descent)
learning_rate = 0.05

for epoch in range(50):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Gradient descent update
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.data:.4f}")
```

---

## Project Structure

- `Value.py` — Core scalar value wrapper supporting DAG construction and `.backward()`.
- `neural_network.py` — Deep learning building blocks (`Module`, `Neuron`, `Layer`, `MLP`).
- `visualize.py` — Computational graph visualization utility (requires `graphviz`).

