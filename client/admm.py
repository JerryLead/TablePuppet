import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from loguru import logger
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from model import Linear, MLP, model_to_vector
from dataset import CustomDataset
import client.base

import ray


@ray.remote(num_gpus=0.2)
class Worker(client.base.Worker):
    def init(self):
        self.m = self.train_data.shape[0]
        self.n = self.train_data.shape[1]
        self.q = self.args.client_num
        self._model = Linear if self.args.model == "Linear" else MLP
        self.models = []
        self.optimizers = []
        self.train_dataloaders = []
        self.test_dataloaders = []
        if self.args.use_DP:
            self.privacy_engines = []
        self.train_data = np.array_split(self.train_data, self.q, axis=0)
        self.test_data = np.array_split(self.test_data, self.q, axis=0)
        for i in range(self.q):
            model = self._model(self.n, self.C).double().to(self.device)
            self.theta_size = model_to_vector(model).size(0)
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.args.local_lr,
                weight_decay=self.args.local_weight_decay,
            )
            train_dataset = CustomDataset(torch.from_numpy(self.train_data[i]))
            train_dataloader = DataLoader(
                train_dataset, batch_size=self.args.local_batch_size, shuffle=False
            )
            test_dataset = CustomDataset(torch.from_numpy(self.test_data[i]))
            test_dataloader = DataLoader(
                test_dataset, batch_size=self.args.local_batch_size, shuffle=False
            )
            if self.args.use_DP:
                privacy_engine = PrivacyEngine(
                    accountant="rdp"
                )  # Default prv, but got numerical calculation error
                (
                    model,
                    optimizer,
                    train_dataloader,
                ) = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_dataloader,
                    # If there is only one client inside an organization, there will be no inside communication round
                    epochs=self.args.local_epoch * self.args.communication_round
                    if self.q == 1
                    else self.args.local_epoch
                    * self.args.communication_round
                    * self.args.inside_communication_round,
                    target_epsilon=self.args.target_epsilon,
                    target_delta=self.args.DP_delta,
                    max_grad_norm=self.args.max_per_sample_clip_norm,
                    poisson_sampling=False,  # Have to align train_data with Y and G, so set posson_sampling to false
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
                    # alphas=list(range(2, 5000)) # UserWarning: Optimal order is the largest alpha. Please consider expanding the range of alphas to get a tighter privacy bound
                )
                logger.info(
                    f"Using sigma={optimizer.noise_multiplier} and C={self.args.max_per_sample_clip_norm}, target epsilon={self.args.target_epsilon}"
                )
                self.privacy_engines.append(privacy_engine)
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.train_dataloaders.append(train_dataloader)
            self.test_dataloaders.append(test_dataloader)

    def _calculate_logits(self, dataloader, model):
        model.eval()
        outputs = []
        with torch.no_grad():
            for batch_T in dataloader:
                batch_T = batch_T.to(self.device)
                outputs.append(model(batch_T).cpu())
        logits = torch.cat(outputs, dim=0)
        return logits.numpy()

    def logits(self, mode="train"):
        dataloaders = (
            self.train_dataloaders if mode == "train" else self.test_dataloaders
        )
        for i in range(self.q):
            logits = [
                self._calculate_logits(dataloaders[i], self.models[i])
                for i in range(self.q)
            ]
        return np.concatenate(logits, axis=0)

    def update_organization_model(self, Y, rho):
        if self.q == 1:
            self.models[0].train()
            for epoch in range(self.args.local_epoch):
                for index, batch_T in enumerate(self.train_dataloaders[0]):
                    self.optimizers[0].zero_grad()
                    batch_G = torch.from_numpy(
                        self.G[
                            index
                            * self.args.local_batch_size : (index + 1)
                            * self.args.local_batch_size
                        ]
                    ).to(self.device)
                    sum_batch_G = torch.sum(batch_G)
                    batch_Y = torch.from_numpy(
                        Y[
                            index
                            * self.args.local_batch_size : (index + 1)
                            * self.args.local_batch_size
                        ]
                    ).to(self.device)
                    batch_T = batch_T.to(self.device)
                    T_x = self.models[0](batch_T)
                    loss = torch.sum(batch_Y * T_x) + rho / 2 * torch.sum(
                        batch_G * torch.norm(T_x, p=2, dim=1) ** 2
                    )
                    loss = loss / sum_batch_G
                    loss.backward()
                    self.optimizers[0].step()
                if self.args.use_DP:
                    logger.debug(
                        f"Table {self.table_name} epoch {epoch}, epsilon = {self.privacy_engines[0].get_epsilon(self.args.DP_delta)}, delta = {self.args.DP_delta}"
                    )
        else:
            Y = np.array_split(Y, self.q, axis=0)
            G = np.array_split(self.G, self.q, axis=0)
            u = np.zeros((self.q, self.theta_size))
            w = np.zeros(self.theta_size)
            theta = np.zeros((self.q, self.theta_size))
            for _ in range(self.args.inside_communication_round):
                for i in range(self.q):
                    theta[i] = self.update_client_model(i, Y[i], G[i], w, u[i], rho)
                theta_bar = np.mean(theta, axis=0)
                u_bar = np.mean(u, axis=0)
                w = theta_bar + u_bar
                u = u + theta - w

    def update_client_model(self, i, Y, G, w, u, rho):
        model = self.models[i]
        optimizer = self.optimizers[i]
        train_dataloader = self.train_dataloaders[i]
        w = torch.from_numpy(w).to(self.device)
        u = torch.from_numpy(u).to(self.device)
        model.train()
        for epoch in range(self.args.local_epoch):
            for index, batch_T in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_G = torch.from_numpy(
                    G[
                        index
                        * self.args.local_batch_size : (index + 1)
                        * self.args.local_batch_size
                    ]
                ).to(self.device)
                sum_batch_G = torch.sum(batch_G)
                batch_Y = torch.from_numpy(
                    Y[
                        index
                        * self.args.local_batch_size : (index + 1)
                        * self.args.local_batch_size
                    ]
                ).to(self.device)
                batch_T = batch_T.to(self.device)
                T_x = model(batch_T)
                theta = model_to_vector(model)
                loss = torch.sum(batch_Y * T_x) + rho / 2 * torch.sum(
                    batch_G * torch.norm(T_x, p=2, dim=1) ** 2
                )
                loss = loss / sum_batch_G + rho / 2 * torch.norm(theta - w + u) ** 2
                loss.backward()
                optimizer.step()
            if self.args.use_DP:
                logger.debug(
                    f"Table {self.table_name} client {i} epoch {epoch}, epsilon = {self.privacy_engines[i].get_epsilon(self.args.DP_delta)}, delta = {self.args.DP_delta}"
                )
        theta = model_to_vector(model).detach().cpu().numpy()
        return theta

    def get_privacy_budget(self):
        if not self.args.use_DP:
            return None
        privacy_budget = 0.0
        for i in range(self.q):
            privacy_budget = max(
                privacy_budget, self.privacy_engines[i].get_epsilon(self.args.DP_delta)
            )
        return privacy_budget
