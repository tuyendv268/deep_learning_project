from datetime import datetime
import logging
import os
from itertools import chain
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ImageDataset
from models.resnet50 import ResNet50
from models.contrastive_loss import ContrastiveLoss
from models.logistic_regression import LogisticRegression


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
    def __init__(self, config):
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.init_model()
        self.init_dataset()
        self.init_loss_optim()
        self.prepare_diretories_and_logger()

        if config['load_contrastive']:
            self.conv_net, self.contrastive_optim = self.load_checkpoint(self.conv_net, self.contrastive_optim, 'checkpoints/conv_net_contrastive.pth')
        if config['load_classification']:
            self.linear = self.load_checkpoint(self.linear, self.classification_optim, 'checkpoints/linear_classification.pth')
            self.conv_net = self.load_checkpoint(self.conv_net, self.classification_optim, 'checkpoints/conv_net_classification.pth')

    def init_model(self):
        logging.info(f'model: {self.config["model"]}')

        if self.config["model"] == "resnet50":
            self.conv_net = ResNet50(self.config['net']).to(self.device)
            
        self.linear = LogisticRegression(self.config['net']).to(self.device)

    def init_dataset(self):
        if self.config["mode"] == "train":
            contrastive_df = pd.read_csv(self.config["contrastive_path"])
            contrastive_df, contrastive_valid_df = train_test_split(
                contrastive_df, 
                test_size=self.config["valid_size"],
                random_state=123,
            )
            contrastive_df = contrastive_df.reset_index()
            contrastive_valid_df = contrastive_valid_df.reset_index()

            self.contrastive_dl = self.get_dataloader(contrastive_df, 'contrastive')
            self.contrastive_valid_dl = self.get_dataloader(contrastive_valid_df)


            train_df = pd.read_csv(self.config["train_path"])
            train_df, valid_df = train_test_split(
                train_df, 
                test_size=self.config["valid_size"],
                random_state=123,
            )

            train_df = train_df.reset_index()
            valid_df = valid_df.reset_index()

            self.train_dl = self.get_dataloader(train_df)
            self.valid_dl = self.get_dataloader(valid_df)

        elif self.config["mode"] == "eval":
            pass

    def init_loss_optim(self):
        self.contrastive_loss = ContrastiveLoss(self.config['net'])
        self.contrastive_optim = optim.Adam(
            params=self.conv_net.parameters(),
            lr=self.config["optim"]["lr"],
            betas=(self.config["optim"]["beta_1"], self.config["optim"]["beta_2"]),
            weight_decay=self.config["optim"]["weight_decay"],

        )
        # self.contrastive_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     self.contrastive_optim,
        #     T_max=self.config['epoch'],
        #     eta_min=self.config['optim']['lr']/50
        # )

        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.classification_optim = optim.Adam(
            params=chain(self.conv_net.parameters(), self.linear.parameters()),
            # params=self.linear.parameters(),
            
            lr=self.config["optim"]["lr"],
            betas=(self.config["optim"]["beta_1"], self.config["optim"]["beta_2"]),
            weight_decay=self.config["optim"]["weight_decay"],
        )
        # self.classification_lr_scheduler = optim.lr_scheduler.MultiStepLR(
        #     self.classification_optim,
        #     milestones=[
        #         int(self.hparams.max_epochs*0.6),
        #         int(self.hparams.max_epochs*0.8)
        #     ],
        #     gamma=0.1
        # )

        self.both_optim = optim.Adam(
            params=chain(self.conv_net.parameters(), self.linear.parameters()),
            lr=self.config["optim"]["lr"],
            betas=(self.config["optim"]["beta_1"], self.config["optim"]["beta_2"])
        )


    def train_contrastive(self):
        logging.info("############# start training contrastive #############")
        
        for epoch in range(self.config["epoch"]):
            train_tqdm = tqdm(self.contrastive_dl, desc=f'epoch={epoch}')
            train_losses = []

            for _, batch in enumerate(train_tqdm):
                self.contrastive_optim.zero_grad()

                images, images_aug, _ = batch
                images = images.to(self.device)
                images_aug = images_aug.to(self.device)

                features = self.conv_net(images, images_aug)

                loss, sim_argsort = self.contrastive_loss(features)
                loss.backward()

                self.contrastive_optim.step()

                train_losses.append(loss.item())
                train_tqdm.set_postfix({
                    "contrastive_loss":np.mean(np.array(train_losses)),
                    
                    'hit_top1':(sim_argsort == 0).float().mean().item(),
                    'hit_top5':(sim_argsort < 5).float().mean().item(),
                    'mean_pos':1+sim_argsort.float().mean().item(),
                })

            if (epoch+1) % self.config["eval_per_n_epochs"] == 0:
                train_loss = np.array(train_losses).mean()
                valid_loss, hit_top1, hit_top5, mean_pos = self.evaluate_contrastive(self.contrastive_valid_dl)

                logging.info(f"validation loss: {valid_loss}") 
                logging.info(f"hit top1: {hit_top1} | hit top5: {hit_top5} | mean pos: {mean_pos}") 

                self.writer.add_scalars('loss_contrastive', {"train":train_loss, "valid":valid_loss}, epoch)
                self.writer.add_scalars('hit', {"top1":hit_top1, 'top5':hit_top5}, epoch)
                self.writer.add_scalars('pos', {"mean":mean_pos}, epoch)

            if (epoch+1) % self.config["save_ckpt_per_n_epochs"] == 0:
                self.save_checkpoint(self.conv_net, self.contrastive_optim, 'checkpoints/conv_net_contrastive.pth', epoch)

        logging.info("training contrastive done!")

    def train_classification(self):
        logging.info("############# start training classification #############")
        
        for epoch in range(self.config["epoch"]):
            train_tqdm = tqdm(self.train_dl, desc=f'epoch={epoch}')
            train_losses = []

            for _, batch in enumerate(train_tqdm):
                self.classification_optim.zero_grad()

                images, images_aug, labels = batch
                images = images.to(self.device)
                # images_aug = images_aug.to(self.device)
                labels = labels.float().squeeze().to(self.device)
                # labels = torch.cat((labels, labels), dim=0)

                features = self.conv_net(images) #images_aug
                preds = self.linear(features)

                loss = self.classification_loss(preds, labels)
                loss.backward()

                self.classification_optim.step()

                train_losses.append(loss.item())
                train_tqdm.set_postfix({
                    "classification_loss":np.mean(np.array(train_losses))
                })
            
            if (epoch+1) % self.config["eval_per_n_epochs"] == 0:
                train_loss = np.array(train_losses).mean()
                valid_results, valid_loss = self.evaluate_classification(self.valid_dl)
                logging.info(f"validation result\n{valid_results}")
                
                self.writer.add_scalars('loss_classification', {"train":train_loss, "valid":valid_loss}, epoch)

            if (epoch+1) % self.config["save_ckpt_per_n_epochs"] == 0:
                self.save_checkpoint(self.linear, self.classification_optim, 'checkpoints/linear_classification.pth', epoch)
                self.save_checkpoint(self.conv_net, self.classification_optim, 'checkpoints/conv_net_classification.pth', epoch)

        logging.info("training classification done!")

    def train_both(self):
        logging.info("############# start training #############")
        
        for epoch in range(self.config["epoch"]):
            train_tqdm = tqdm(self.train_dl, desc=f'epoch={epoch}')
            contrastive_losses = []
            classification_losses = []

            for _, batch in enumerate(train_tqdm):
                self.both_optim.zero_grad()

                images, images_aug, labels = batch
                images = images.to(self.device)
                images_aug = images_aug.to(self.device)
                labels = labels.float().squeeze().to(self.device)
                labels = torch.cat((labels, labels), dim=0)

                features = self.conv_net(images, images_aug)
                preds = self.linear(features)

                loss_contrastive, sim_argsort = self.contrastive_loss(features)
                loss_classification = self.classification_loss(preds, labels)
                loss = loss_contrastive + loss_classification
                loss.backward()

                self.both_optim.step()

                contrastive_losses.append(loss_contrastive.item())
                classification_losses.append(loss_classification.item())

                train_tqdm.set_postfix({
                    "contrastive_loss":np.mean(np.array(contrastive_losses)),
                    "classification_loss":np.mean(np.array(classification_losses)),
                    'hit_top1':(sim_argsort == 0).float().mean().item(),
                    'hit_top5':(sim_argsort < 5).float().mean().item(),
                    'mean_pos':1+sim_argsort.float().mean().item(),
                })


            if (epoch+1) % self.config["eval_per_n_epochs"] == 0:
                mean_loss_contrastive = np.array(contrastive_losses).mean()
                valid_loss_contrastive, hit_top1, hit_top5, mean_pos = self.evaluate_contrastive(self.valid_dl)

                logging.info(f"validation contrastive loss: {valid_loss_contrastive}") 
                logging.info(f"hit top1: {hit_top1} | hit top5: {hit_top5} | mean pos: {mean_pos}") 
                self.writer.add_scalars('loss_contrastive', {"train":mean_loss_contrastive, "valid":valid_loss_contrastive}, epoch)
                self.writer.add_scalars('hit', {"top1":hit_top1, 'top5':hit_top5}, epoch)
                self.writer.add_scalars('pos', {"mean":mean_pos}, epoch)


                mean_loss_classification = np.array(classification_losses).mean()
                valid_results, valid_loss_classification = self.evaluate_classification(self.valid_dl)

                logging.info(f"validation classification loss: {valid_loss_classification}") 
                logging.info(f"validation result\n{valid_results}")
                self.writer.add_scalars('loss_classification', {"train":mean_loss_classification, "valid":valid_loss_classification}, epoch)

            if (epoch+1) % self.config["save_ckpt_per_n_epochs"] == 0:
                self.save_checkpoint(self.linear, self.classification_optim, 'checkpoints/linear_both.pth', epoch)
                self.save_checkpoint(self.conv_net, self.classification_optim, 'checkpoints/conv_net_both.pth', epoch)

        logging.info("training done!")



    @torch.no_grad()
    def evaluate_contrastive(self, dataloader):
        losses = []

        all_hit_top1 = []
        all_hit_top5 = []
        all_mean_pos = []

        for i, batch in enumerate(dataloader):
            images, images_aug, _ = batch
            images = images.to(self.device)
            images_aug = images_aug.to(self.device)
            
            features = self.conv_net(images, images_aug)

            loss, sim_argsort = self.contrastive_loss(features)
            losses.append(loss.item())

            all_hit_top1.extend((sim_argsort == 0).float().tolist())
            all_hit_top5.extend((sim_argsort < 5).float().tolist())
            all_mean_pos.extend((1+sim_argsort).float().tolist())

        mean_loss = np.array(losses).mean()
        mean_hit_top1 = np.array(all_hit_top1).mean()
        mean_hit_top5 = np.array(all_hit_top5).mean()
        mean_pos = np.array(all_mean_pos).mean()

        return mean_loss, mean_hit_top1, mean_hit_top5, mean_pos

    @torch.no_grad()
    def evaluate_classification(self, dataloader):
        y_trues, y_predicts = [], []
        losses = []

        for i, batch in enumerate(dataloader):
            images, images_aug, labels = batch
            images = images.to(self.device)
            images_aug = images_aug.to(self.device)
            labels = labels.float().squeeze().to(self.device)
            labels = torch.cat((labels, labels), dim=0)

            features = self.conv_net(images, images_aug)
            preds = self.linear(features)

            loss = self.classification_loss(preds, labels)
            losses.append(loss.item())

            y_trues.extend(labels.tolist())
            y_predicts.extend(torch.round(preds).tolist())

        y_trues = np.array(y_trues)
        y_predicts = np.array(y_predicts)
    
        cls_results = classification_report(y_trues, y_predicts, target_names=self.config["dataset"]["labels"], zero_division=0)
        return cls_results, np.array(losses).mean()


    def get_dataloader(self, df, mode=None):
        img_dataset = ImageDataset(
            datas=df,
            config=self.config["dataset"],
            mode=mode
        )

        img_dataloader = DataLoader(
            img_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )
        
        return img_dataloader

    def save_checkpoint(self, model, optimizer, path, epoch):
        state_dict = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": epoch}
        
        torch.save(state_dict, path)
        logging.info(f'saved model and optimizer state dict to {path}')
        
        
    def load_checkpoint(self, model, optimizer, path):
        state_dict = torch.load(path, map_location=self.device)
        model_state_dict = state_dict["model_state_dict"]
        optim_state_dict = state_dict["optim_state_dict"]
        
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optim_state_dict)
        
        logging.info(f'load model and optimizer state dict from {path}')
        return model, optimizer

    def prepare_diretories_and_logger(self):
        current_time = datetime.now()
        current_time = current_time.strftime("%d-%m-%Y_%H-%M-%S")

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