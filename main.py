import yaml
from yaml.loader import SafeLoader
from src.trainer import Trainer

if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
        
    trainer = Trainer(config)
    trainer.train()