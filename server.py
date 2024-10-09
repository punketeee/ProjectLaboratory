from collections import OrderedDict
import fcntl
import json
import os
from typing import Dict, Tuple, Any
from flwr.common import NDArrays, Scalar
from omegaconf import DictConfig
import torch
import flwr as fl
from flwr.common import Context
from flwr.server import Driver, LegacyContext, ServerConfig
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from model import Net, test
from dataset import prepare_dataset
from client import generate_client_fn
from omegaconf import DictConfig

import torch

from model import Net, test


def get_on_fit_config(config: DictConfig, num_clients: int):
    """Return function that prepares config to send to clients."""


    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.
        #server_run input argument is in case settins of clients should be changed during the FL process

        return {
            "lr": config.lr,
            "momentum": config.momentum,
            "local_epochs": config.local_epochs,
            "num_clients": num_clients,
            "round": server_round,
        }

    return fit_config_fn

def save_evaluation_results(results, filename):
    """Save evaluation results to a JSON file."""
    filepath = os.path.join("results", filename)
    if not os.path.exists("results"):
        os.makedirs("results")
    
    lockfile = f"{filepath}.lock"

    with open(lockfile, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        if os.path.exists(filepath):
            if os.path.getsize(filepath) > 0:
                with open(filepath, "r") as file:
                    data = json.load(file)
            else:
                data = []
            data.append(results)
        else:
            data = [results]

        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)

        fcntl.flock(lock, fcntl.LOCK_UN)

def get_evaluate_fn(num_classes: int, global_testloader, num_clients: int, evaluation_mode: str):
    """Define function for global evaluation on the server."""
    def evaluate_fn(server_round: int, parameters, config):
        # This function is called by the strategy's `evaluate()` method.

        print(f"Evaluating in round {server_round}")
        """Evaluate the global model and return scores."""
        model = Net(num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load global parameters into the server model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        config["num_clients"] = num_clients

        if evaluation_mode == "global":
            loss, accuracy = test(model, global_testloader, device)
        else:
            # In local mode, centralized evaluation might not be performed
            loss, accuracy = 0.0, 0.0

        
        # Save evaluation results
        eval_results = {
            "round": server_round,
            "loss": loss,
            "accuracy": accuracy,
            "num_clients": num_clients
        }
        print(f"Global Evaluation results (round {server_round}): {eval_results}")
        save_evaluation_results(eval_results, "global_eval_results.json")

     
        # Include the global accuracy in the evaluation config for client use
        config["global_average_accuracy"] = accuracy

       # print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")

        return loss, {"accuracy": accuracy, "num_clients": num_clients}

    return evaluate_fn

def get_on_evaluate_config(num_clients: int):
    def evaluate_config_fn(server_round: int):
        return {
            "num_clients": num_clients,
            "round": server_round,
        }
    return evaluate_config_fn


def initialize_server(cfg: DictConfig, trainloaders, validationloaders, local_testloaders, global_testloader):

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=1.0,
        min_evaluate_clients=cfg.num_clients_per_round_eval,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit, cfg.num_clients),
        on_evaluate_config_fn=get_on_evaluate_config(cfg.num_clients),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, global_testloader, cfg.num_clients, cfg.evaluation_mode),
    )
    client_fn = generate_client_fn(trainloaders, validationloaders, local_testloaders, global_testloader, cfg.num_classes, cfg.evaluation_mode, cfg.compute_l1o, cfg.compute_i1i)

    app = fl.server.ServerApp()

    @app.main()
    def main(driver: Driver, context: Context) -> None:
        context = LegacyContext(
            state=context.state,
            config=ServerConfig(num_rounds=cfg.num_rounds),
            strategy=strategy,
        )
        workflow = DefaultWorkflow(
            fit_workflow=SecAggPlusWorkflow(
                num_shares=cfg.secagg.num_shares,
                reconstruction_threshold=cfg.secagg.reconstruction_threshold,
                timeout=cfg.secagg.timeout,
            )
        )
        workflow(driver, context)

    return app, client_fn