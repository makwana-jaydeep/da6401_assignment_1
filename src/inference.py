import argparse
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on test set")

    # Same CLI as train.py with best config as default values
    parser.add_argument("-d", "--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-o", "--optimizer", type=str, default="rmsprop",
                        choices=["sgd", "momentum", "nag"])
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers", type=int, default=3)
    parser.add_argument("-sz", "--hidden_size", type=int, nargs="+", default=[128, 128, 128])
    parser.add_argument("-a", "--activation", type=str, default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("-w_p", "--wandb_project", type=str, default="da6401_assignment1")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--model_path", type=str, default="src/best_model.npy")

    return parser.parse_args()


def load_model(model_path):
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    metrics = model.evaluate(X_test, y_test)
    return {
        "logits": metrics["logits"],
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
    }

# Main function to run inference, loading model and data, and print the  results
def main():
    args = parse_arguments()

    if len(args.hidden_size) < args.num_layers:
        args.hidden_size = args.hidden_size + [args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size))
    elif len(args.hidden_size) > args.num_layers:
        args.hidden_size = args.hidden_size[:args.num_layers]

    print(f"Loading dataset: {args.dataset}")
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"F1-Score  : {results['f1']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"Loss      : {results['loss']:.4f}")

    return results


if __name__ == "__main__":
    main()
