from collections import OrderedDict
import fcntl
import os
import json
from typing import Dict, Tuple, Any
from flwr.common import NDArrays, Scalar
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
import torch
import flwr as fl
from model import Net, train, test


class FlowerClient(fl.client.NumPyClient):
    """Define a Flower Client with I1I and L1O scoring."""

    def __init__(self, trainloader, valloader, local_testloader, global_testloader, num_classes, eval_mode, compute_l1o, compute_i1i, client_id) -> None:
        super().__init__()

        # Dataloaders associated with this client
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_testloader = local_testloader
        self.global_testloader = global_testloader

        # Randomly initialized model
        self.model = Net(num_classes)

        # Device determination
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Store initial global model parameters
        self.initial_parameters = None
        self.local_parameters = self.get_parameters({})
        self.eval_mode = eval_mode

        self.compute_l1o = compute_l1o
        self.compute_i1i = compute_i1i
        self.client_id = client_id

        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Receive parameters and apply them to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar] = {}) -> NDArrays:
        """Extract model parameters and return them as a list of numpy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train model using local data and compute I1I score."""
        # Copy global model parameters
        self.set_parameters(parameters)

        # Evaluate the global model before local training (for I1I calculation)
        if self.compute_i1i:
            _, acc_before = test(self.model, self.valloader, self.device)

        # Fetch training hyperparameters
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        

        # Initialize optimizer and train the local model
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # train model
        train(self.model, self.trainloader, optimizer, epochs, self.device)

        if self.compute_i1i:
            _, acc_after = test(self.model, self.valloader, self.device)
            i1i_score = acc_after - acc_before
          #  print(f"I1I Score (Client): {i1i_score}")
          #  print(f"Training Accuracy: Before={acc_before}, After={acc_after}")
        else:
            i1i_score = 0.0

        # Save the locally trained parameters to use them later for L1O
        self.local_parameters = self.get_parameters({})

         # Save fit results
        fit_results = {
            "client_id": self.client_id,
            "round": config.get('round', 0),
           # "round": round_num,
            "i1i_score": i1i_score
        }
        print(f"Fit results (round {config.get('round', 0)}): {fit_results}")
        self._save_results(fit_results, "fit_results.json")

        # Return the updated model and the I1I score
        return self.get_parameters({}), len(self.trainloader), {"i1i_score": i1i_score}

    def evaluate(self, parameters, config):
        """Evaluate the model and compute the L1O score."""
        # Set parameters to the global model
        
        self.set_parameters(parameters)
     

        if self.eval_mode == "global":
            _, global_accuracy = test(self.model, self.valloader, self.device)
        else:  # local evaluation
            _, global_accuracy = test(self.model, self.local_testloader, self.device)


        # Get the number of clients
        num_clients = config.get("num_clients", 5)

        if self.compute_l1o:
        # Adjust the global model to exclude this client's contribution
            local_state_dict = OrderedDict(zip(self.model.state_dict().keys(), 
                                       [torch.tensor(np_param) for np_param in self.local_parameters]))
        # Clone the model to avoid in-place modification issues
            temp_model = Net(10).to(self.device)
            temp_model.load_state_dict(self.model.state_dict())

         # Remove the contribution of the local model from the global model
            for key in temp_model.state_dict().keys():
                if key in local_state_dict:
                    original_value = temp_model.state_dict()[key].clone()
                    contribution_value = local_state_dict[key] / num_clients
                    temp_model.state_dict()[key] -= contribution_value
                    
            if self.eval_mode == "global":
                _, l1o_accuracy = test(temp_model, self.global_testloader, self.device)
            else:
                _, l1o_accuracy = test(temp_model, self.local_testloader, self.device)
          #  print(f"L1O accuracy: {l1o_accuracy}")
            # Calculate the L1O score as the difference in accuracies
            l1o_score = global_accuracy - l1o_accuracy
            # Print the L1O score for analysis purposes
           # print(f"L1O Score (Client): {l1o_score}")
        else:
            l1o_score = 0.0

         # Save evaluation results
        eval_results = {
            "client_id": self.client_id,
            "round": config.get('round', 0),
            #  "round": config.get("server_round", 0),
            "global_accuracy": global_accuracy,
            "l1o_score": l1o_score
        }
        print(f"Evaluation results (round {config.get('round', 0)}): {eval_results}")
        self._save_results(eval_results, "eval_results.json")    
        # Return the global model's performance along with the L1O score
        return float(global_accuracy), len(self.valloader), {"accuracy": global_accuracy, "l1o_score": l1o_score, "num_clients": num_clients}

    def _save_results(self, results, filename):
        filepath = os.path.join(self.results_dir, filename)
        lockfile = f"{filepath}.lock"

        with open(lockfile, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            try:
                # Read existing data
                if os.path.exists(filepath):
                    if os.path.getsize(filepath) > 0:
                        with open(filepath, "r") as file:
                            try:
                                data = json.load(file)
                            except json.JSONDecodeError:
                                data = []  # Reset data if JSON is corrupted
                    else:
                        data = []
                else:
                    data = []

                # Append new results
                data.append(results)

                # Write updated data back to file
                with open(filepath, "w") as file:
                    json.dump(data, file, indent=4)

            except Exception as e:
                print(f"Failed to save results: {e}")

            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

def generate_client_fn(trainloaders, valloaders, local_testloaders, global_testloader, num_classes, eval_mode, compute_l1o, compute_i1i):
    """Return a function that can be used by the VirtualClientEngine."""

    def client_fn(cid: str) -> fl.client.NumPyClient:
        # This function will be called internally by the VirtualClientEngine
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            local_testloader=local_testloaders[int(cid)],
            global_testloader=global_testloader,
            num_classes=num_classes,
            eval_mode=eval_mode,
            compute_l1o=compute_l1o,
            compute_i1i=compute_i1i,
            client_id=cid
        ).to_client()

    return client_fn

app = ClientApp(
    client_fn=generate_client_fn,
    mods=[
        secaggplus_mod,
    ],
)