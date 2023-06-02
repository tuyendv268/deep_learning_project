import yaml
from yaml.loader import SafeLoader
from trainer import Trainer

if __name__ == "__main__":
    with open("config.yml", "r") as f:
        config = yaml.load(f, Loader=SafeLoader)
        
    trainer = Trainer(config)

    trainer.train_contrastive()
    trainer.train_classification()
    
    # trainer.train_both()