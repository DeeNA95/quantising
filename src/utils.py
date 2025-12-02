import yaml
import wandb


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config, experiment_id):
    """
    Initialize WandB run.
    """
    run = wandb.init(
        project=config["project"]["name"],
        entity=config["project"].get("entity"),
        name=f"{experiment_id}-{config['experiments'][experiment_id]['description']}",
        config=config["experiments"][experiment_id],
    )
    return run
