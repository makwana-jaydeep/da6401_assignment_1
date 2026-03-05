import numpy as np
from types import SimpleNamespace
from ann.neural_network import NeuralNetwork
from ann.objective_functions import LOSS_FN

# grad check for first layer
np.random.seed(42)
args = SimpleNamespace(
    dataset='mnist', epochs=1, batch_size=5, loss='cross_entropy',
    optimizer='sgd', learning_rate=0.0, weight_decay=0.0,
    num_layers=2, hidden_size=[8, 8], activation='tanh',
    weight_init='xavier', wandb_project='test', wandb_entity=None
)

# create model based on CLI args
model = NeuralNetwork(args)
X = np.random.randn(5, 784) * 0.1
y = np.array([0, 1, 2, 3, 4])
logits = model.forward(X)
model.backward(y, logits)
analytical = model.layers[0].grad_W.copy()
eps = 1e-5
numerical = np.zeros_like(model.layers[0].W)

# compute grads
for i in range(5):
    for j in range(5):
        
        model.layers[0].W[i,j] += eps
        lp = LOSS_FN['cross_entropy'](model.forward(X), y)
        model.layers[0].W[i,j] -= 2*eps
        lm = LOSS_FN['cross_entropy'](model.forward(X), y)
        model.layers[0].W[i,j] += eps
        numerical[i,j] = (lp - lm) / (2*eps)

diff = np.max(np.abs(analytical[:5,:5] - numerical[:5,:5]))

print(f'Max gradient difference: {diff:.2e}')
print('PASS' if diff < 1e-7 else 'FAIL')