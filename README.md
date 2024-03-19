# Advanced PyTorch Optimizer: SophiaOptim

### Overview

This project introduces SophiaOptim, my take on the Sophia (Second-order Clipped Stochastic Optimization) optimizer. A custom PyTorch optimizer designed to enhance the training performance of deep learning models, SophiaOptim integrates sophisticated optimization techniques, including Lookahead mechanics and Exponential Moving Average (EMA) of model weights, alongside adaptive learning rates and Hessian-based updates for efficient and robust model optimization.

### Features

Lookahead Mechanism: Implements the Lookahead optimization strategy to maintain two sets of weights, providing a more stable and consistent training process.
Exponential Moving Average (EMA): Utilizes EMA of model weights for smoother optimization, leveraging the averaged weights for evaluation to achieve better generalization performance.

Adaptive Learning Rate: Adapts learning rates based on the Hessian diagonal's estimation, allowing for more informed update steps.
Hutchinson’s Hessian Estimation: Estimates the Hessian diagonal efficiently using Hutchinson's method, incorporating second-order information without the computational overhead of full Hessian computation.

### Installation

To use SophiaOptim in your project, ensure you have PyTorch installed. Clone this repository and import SophiaOptim and Lookahead into your training script:


`
git clone https://github.com/alonso130r/SOPHIA-optimizer.git
`

```
from sophia_optim import SophiaOptim
from lookahead import Lookahead
```

### Usage

To integrate SophiaOptim into your training loop, initialize the optimizer with your model's parameters and specify any desired configurations:

```
optimizer = SophiaOptim(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, rho=0.1, weight_decay=0.01, ema_decay=0.999)
```

#### Use the optimizer in your training loop as you would with any standard PyTorch optimizer. Remember to update the EMA weights and apply them for model evaluation.
