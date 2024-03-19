from sophia_optim import SophiaOptim
from lookahead import Lookahead

"""
model = ...   (Your model)
base_optimizer = SophiaOptim(model=model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, rho=0.1, weight_decay=0.01)
lookahead_optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)

# In your training loop:
for input, target in dataset:
    lookahead_optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    lookahead_optimizer.step()
"""