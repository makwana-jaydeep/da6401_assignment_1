"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

import numpy as np
from ann.neural_layer import Layer
from ann.activations import softmax
from ann.objective_functions import LOSS_FN, LOSS_GRAD
from ann.optimizers import OPTIMIZERS
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class NeuralNetwork:
    def __init__(self, cli_args):
        self.args = cli_args
        self.layers = []
        self._build()
        self.optimizer = OPTIMIZERS[cli_args.optimizer](
            lr=cli_args.learning_rate,
            weight_decay=cli_args.weight_decay,
        )

        self.optimizer.init_state(self.layers)

    # building the nn architecture
    def _build(self):

        a = self.args
        num_layers = getattr(a, 'num_layers', 3)
        hidden_size = getattr(a, 'hidden_size', [128] * (num_layers - 1))

        hidden_sizes = hidden_size if isinstance(hidden_size, list) else [hidden_size] * (num_layers - 1)
        
        num_hidden = num_layers

        if len(hidden_sizes) < num_hidden:
            hidden_sizes = hidden_sizes + [hidden_sizes[-1]] * (num_hidden - len(hidden_sizes))

        elif len(hidden_sizes) > num_hidden:
            hidden_sizes = hidden_sizes[:num_hidden]

        dims = [784] + hidden_sizes + [10]

        # dim will have num_layers connections exactly
        activation = getattr(a, 'activation', 'relu')
        weight_init = getattr(a, 'weight_init', 'xavier')

        for i in range(len(dims) - 1):
            act = activation if i < len(dims) - 2 else None
            self.layers.append(Layer(dims[i], dims[i + 1], act, weight_init))

    # forward pass
    def forward(self, X):
        out = X

        for layer in self.layers:
            out = layer.forward(out)

        return out  # logits

    # backward pass
    def backward(self, y_true, y_pred):

        # gradient of loss wrt logits
        delta = LOSS_GRAD[self.args.loss](y_pred, y_true)
        grads_w, grads_b = [], []

        for layer in reversed(self.layers):
            delta = layer.backward(delta)
            grads_w.insert(0, layer.grad_W)
            grads_b.insert(0, layer.grad_b)

        return grads_w, grads_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    # training loop
    def train(self, X_train, y_train, epochs, batch_size, X_val=None, y_val=None, wandb_run=None):

        n = X_train.shape[0]
        best_f1 = -1
        best_weights = None

        for epoch in range(epochs):

            idx = np.random.permutation(n)
            X_train, y_train = X_train[idx], y_train[idx]
            epoch_loss = 0.0

            for start in range(0, n, batch_size):

                Xb = X_train[start:start + batch_size]
                yb = y_train[start:start + batch_size]
                logits = self.forward(Xb)
                loss = LOSS_FN[self.args.loss](logits, yb)
                epoch_loss += loss * len(yb)
                self.backward(yb, logits)
                self.update_weights()

            epoch_loss /= n
            train_metrics = self.evaluate(X_train, y_train)
            log = {
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "train_acc": train_metrics["accuracy"],
            }

            if X_val is not None:

                val_metrics = self.evaluate(X_val, y_val)
                log["val_loss"]  = val_metrics["loss"]
                log["val_acc"] = val_metrics["accuracy"]
                log["val_f1"] = val_metrics["f1"]
                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    best_weights = self.get_weights()

            if wandb_run is not None:
                wandb_run.log(log)

            print(f"Epoch {epoch+1}/{epochs} | loss: {epoch_loss:.4f} | train_acc: {train_metrics['accuracy']:.4f}", end="")

            if X_val is not None:
                print(f" | val_acc: {log['val_acc']:.4f}", end="")

            print()

        return best_weights

    # evalution on data
    def evaluate(self, X, y):

        logits = self.forward(X)
        loss = LOSS_FN[self.args.loss](logits, y)
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        prec = precision_score(y, preds, average="macro", zero_division=0)
        rec = recall_score(y, preds, average="macro", zero_division=0)

        return {"loss": loss, "accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "logits": logits}

    def get_weights(self):

        weights = {}
        for i, l in enumerate(self.layers):
            weights[f"W{i}"] = l.W.copy()
            weights[f"b{i}"] = l.b.copy()
        return weights

    def set_weights(self, weights):

        for i, layer in enumerate(self.layers):

            if f"W{i}" in weights:
                # flat format
                layer.W = weights[f"W{i}"].copy()
                layer.b = weights[f"b{i}"].copy()

            elif str(i) in weights:
                # nested string format
                layer.W = weights[str(i)]["W"].copy()
                layer.b = weights[str(i)]["b"].copy()

            elif i in weights:
                # nested intger format
                layer.W = weights[i]["W"].copy()
                layer.b = weights[i]["b"].copy()
            else:
                raise KeyError(f"Layer {i} not found. Keys: {list(weights.keys())}")
