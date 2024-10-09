
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pickle
from pathlib import Path
import flwr as fl
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evaluate_fn
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pickle
from pathlib import Path
from server import initialize_server
import flwr as fl
from flwr.client.mod import secaggplus_mod


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    num_clients = cfg.num_clients

    trainloaders, validationloaders, local_testloaders, global_testloader = prepare_dataset(
        num_clients, cfg.batch_size
    )

    app, client_fn = initialize_server(cfg, trainloaders, validationloaders, local_testloaders, global_testloader)

    client_app = fl.client.ClientApp(client_fn, mods=[
        secaggplus_mod,
    ],)

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # Run the simulation
    fl.simulation.run_simulation(
        server_app=app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,  
        enable_tf_gpu_growth=False,
        verbose_logging=False
    )

if __name__ == "__main__":
    main()