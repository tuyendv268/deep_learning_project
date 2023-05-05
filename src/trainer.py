import torch
from torch.utils.data import DataLoader
from src.models.resnet18 import ResNet18
from src.dataset import ImageDataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from src.utils import load_data
import numpy as np
from tqdm import tqdm

class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        
        self.init_model()
        self.init_optim()
        
        self.loss_func = CrossEntropyLoss()
        
        if self.config["mode"] == "train":
            train_df = load_data(self.config["train_path"])
            test_df = load_data(self.config["test_path"])
            valid_df = load_data(self.config["valid_path"])
            
            self.train_dl = self.prepare_dataloader(train_df)
            self.valid_dl = self.prepare_dataloader(valid_df)
            self.test_dl = self.prepare_dataloader(test_df)

    def prepare_dataloader(self, datas):
        img_dataset = ImageDataset(
            datas=datas,
            config=self.config["dataset"])
        
        img_dataloader = DataLoader(
            img_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )
        
        return img_dataloader
    
    def init_model(self):
        if self.config["model"] == "resnet18":
            self.model = ResNet18()
            
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        
        print(self.model)
        print("num param: ", params)
    
    def init_optim(self):
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.config["optim"]["lr"],
            betas=(self.config["optim"]["beta_1"], self.config["optim"]["beta_2"])
        )
        
    def train(self):
        print("############# start training #############")
        
        for epoch in range(self.config["epoch"]):
            train_tqdm = tqdm(self.train_dl, desc=f"epoch={epoch}")
            train_losses = []
            for i, batch in enumerate(train_tqdm):
                self.optimizer.zero_grad()
                images, labels = batch
                preds = self.model(images)
                
                preds = preds.reshape(preds.size(0) * preds.size(1), -1)
                labels = labels.reshape(labels.size(0) * labels.size(1))
                
                loss = self.loss_func(preds, labels)
                loss.backward()
                
                self.optimizer.step()
                
                train_losses.append(loss.item())
                train_tqdm.set_postfix({
                    "loss":np.mean(np.array(train_losses))
                })