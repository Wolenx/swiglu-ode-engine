# SwiGLU-ODE: Custom Autodiff Engine for Continuous Dynamics

This repository implements a pure-JAX neural network engine designed to parameterize continuous-time physical dynamical systems (Neural ODEs) using modern high-dimensional LLM primitives. 

Unlike standard implementations that rely on framework-level automatic differentiation (e.g., `jax.value_and_grad`), this codebase features manually derived and implemented Vector-Jacobian Products (VJPs) for SwiGLU activations and RMSNorm layers. The entire forward, backward, and time-integration passes are compiled into $O(1)$ XLA graphs using `jax.lax.scan`.

## Architecture & Mathematical Implementation

* **Manual Reverse-Mode Autodiff:** Exact mathematical derivation of the backward pass for SwiGLU and RMSNorm. Gradients are verified against XLA's internal autograd engine via finite differences and exact analytical comparisons.
* **Continuous-Domain Normalization:** The VJP implementation for the RMSNorm layer incorporates the exact analytical derivative to prevent magnitude collapse in continuous phase-space regimes.
* **$O(1)$ Graph Compilation:** Network depth and training epochs are unrolled via `jax.lax.scan`, ensuring compilation time remains constant regardless of architecture depth.
* **Pure Functional State:** The architecture avoids class mutation by registering as a JAX PyTree (`tree_flatten`, `tree_unflatten`), allowing full training loops to execute as a single GPU dispatch.

The exact analytical derivative implemented for the RMSNorm backward pass is:

$$\frac{\partial y}{\partial x} = \frac{h}{r + \epsilon} - x \left( \frac{x \cdot h}{N \cdot r \cdot (r + \epsilon)^2} \right)$$

where $r = \sqrt{\frac{1}{N} \sum x^2}$.

## Repository Structure

```text
swiglu-ode/
├── pyproject.toml
├── README.md
├── src/
│   └── swiglu_ode/
│       ├── __init__.py
│       ├── activations.py
│       ├── normalization.py
│       └── network.py
├── experiments/
│   └── 01_harmonic_oscillator.py
└── tests/
    └── test_gradients.py
```

## Installation

This project is packaged for modular importing. It requires Python 3.10+ and JAX. Clone the repository and install it in editable mode:

```bash
git clone [https://github.com/Wolenx/swiglu-ode.git](https://github.com/Wolenx/swiglu-ode.git)
cd swiglu-ode
pip install -e .
```

## Verification & Benchmarks

### 1. Autodiff Verification
To verify the manually derived VJPs against JAX's internal autograd engine:

```bash
python tests/test_gradients.py
```
*Expected output: Silent pass or `VJP Verification Passed: Manual gradients exactly match XLA Autograd.`*

### 2. Continuous-Time Integration Benchmark
To test the architecture's capacity for modeling physical systems, the network is trained to learn the 2D vector field of a damped harmonic oscillator from normalized phase-space coordinates.

```bash
python experiments/01_harmonic_oscillator.py
```

**Integration Rollout Performance:**
Using standard gradient descent on the custom backward pass, the architecture achieves a stable linear drift error over a 35-step Euler integration rollout. 

```text
Compiling XLA graph and starting training...
Epoch 0000 | Vector Field MSE: 0.523154
Epoch 0300 | Vector Field MSE: 0.003217
Epoch 0600 | Vector Field MSE: 0.001412
Epoch 0900 | Vector Field MSE: 0.000820
Epoch 1200 | Vector Field MSE: 0.000556
Epoch 1499 | Vector Field MSE: 0.000415

Integrating learned vector field over time...
Step   | X_pred   | Y_pred   | Drift Error
----------------------------------------
0      |  1.500   |  0.000   | 0.000376
5      |  1.463   | -0.354   | 0.000036
10     |  1.344   | -0.649   | 0.000426
15     |  1.160   | -0.873   | 0.000748
20     |  0.925   | -1.019   | 0.000890
25     |  0.658   | -1.089   | 0.001408
30     |  0.379   | -1.085   | 0.001500
35     |  0.110   | -1.013   | 0.001290
```
