from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.models.resnet18 import ResNet18
from src.models.restnet50 import ResNet50
from src.dataset import ImageDataset
from torch.nn import CrossEntropyLoss
from src.utils import load_data
from datetime import datetime
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import logging
import torch
import os

current_time = datetime.now()
current_time = current_time.strftime("%d-%m-%Y_%H-%M-%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/log_{current_time}.log"),
        logging.StreamHandler()
    ]
)

class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        
        self.init_model()
        self.init_optim()
        self.prepare_diretories_and_logger()
        
        self.loss_func = CrossEntropyLoss()
        
        if self.config["mode"] == "train":
            train_df = load_data(self.config["train_path"])
            test_df = load_data(self.config["test_path"])
            # valid_df = load_data(self.config["valid_path"])
            train_df, valid_df = train_test_split(train_df, test_size=self.config["valid_size"], random_state=42)
            train_df, valid_df = train_df.reset_index(), valid_df.reset_index()
            
            logging.info("prepare data for train, valid, test")
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
            drop_last=False)
        
        return img_dataloader
    
    def init_model(self):
        logging.info(f'model: {self.config["model"]}')
        if self.config["model"] == "resnet18":
            self.model = ResNet18()
        elif self.config["model"] == "resnet50":
            self.model = ResNet50()
            
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f"num params: {params}")
            
    def init_optim(self):
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.config["optim"]["lr"],
            betas=(self.config["optim"]["beta_1"], self.config["optim"]["beta_2"])
        )
        
    def save_checkpoint(self, path, epoch):
        state_dict = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "epoch": epoch}
        
        torch.save(state_dict, path)
        logging.info(f'saved model and optimizer state dict to {path}')
        
    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location="cpu")
        model_state_dict = state_dict["model_state_dict"]
        optim_state_dict = state_dict["optim_state_dict"]
        
        self.model.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optim_state_dict)
        
        logging.info(f'load model and optimizer state dict from {path}')
    
    def prepare_diretories_and_logger(self):
        current_time = datetime.now()
        current_time = current_time.strftime("%d-%m-%Y_%H:%M:%S")

        log_dir = f"{self.config['log_dir']}/{current_time}"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            logging.info(f"logging into {log_dir}")
            
        checkpoint_dir = self.config["ckpt_dir"]
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
            logging.info(f'mkdir {checkpoint_dir}')
        
        self.writer = SummaryWriter(
            log_dir=log_dir
        )
        
    def train(self):
        logging.info("############# start training #############")
        
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
            if (epoch+1) % self.config["save_ckpt_per_n_epochs"] == 0:
                train_loss = np.array(train_losses).mean()
                valid_results, valid_loss = self.evaluate(self.model, self.valid_dl)
                logging.info(f"validation result\n{valid_results}")
                
                self.writer.add_scalars('loss', {"train":train_loss, "valid":valid_loss}, epoch)
        
        logging.info("Done!")
                
    @torch.no_grad()
    def evaluate(self, model, dataloader):
        y_trues, y_predicts = [], []
        losses = []
        for i, batch in enumerate(dataloader):
            images, labels = batch
            preds = model(images)
            
            _preds = preds.reshape(preds.size(0) * preds.size(1), -1)
            _labels = labels.reshape(labels.size(0) * labels.size(1))
            
            loss = self.loss_func(_preds, _labels)
            
            losses.append(loss.item())
            y_trues += labels.squeeze(-1).tolist()
            y_predicts += torch.argmax(preds, dim=2).tolist()
        
        y_trues = np.array(y_trues)
        y_predicts = np.array(y_predicts)
    
        cls_results = classification_report(y_trues, y_predicts, target_names=self.config["dataset"]["labels"], zero_division=0)
        return cls_results, np.array(losses).mean()