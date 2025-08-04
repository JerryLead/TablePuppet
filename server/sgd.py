import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torcheval.metrics import BinaryAUROC
import math
import server.base
import numpy as np

from model import Linear, MLP

from util.ray_utils import ray_group_call, ray_group_call_multiargs

import ray


class Server(server.base.Server):
    def init(self):
        self.loss = (
            torch.nn.CrossEntropyLoss()
            if self.task == "classification"
            else torch.nn.MSELoss()
        )
        self.G = self.aligned_G.copy()
        self.test_G = self.aligned_test_G.copy()

        align_args = [
            (self.G[i], self.test_G[i], self.f[:, i + 1], self.test_f[:, i + 1])
            for i in range(len(self.workers))
        ]
        align_results = ray_group_call_multiargs(self.workers, "align", align_args)
        self.G, self.test_G = zip(*align_results)
        self.G = list(self.G)
        self.test_G = list(self.test_G)
        ray_group_call(self.workers, "init")
        self.train_iterations = math.ceil(self.M / self.args.batch_size)
        self.test_iterations = math.ceil(self.test_M / self.args.batch_size)
        if isinstance(self.b, np.ndarray):
            self.b = torch.from_numpy(self.b)
        if isinstance(self.test_b, np.ndarray):
            self.test_b = torch.from_numpy(self.test_b)
        if self.task == "regression":
            self.b, self.test_b = self.b.double(), self.test_b.double()
        # self.b, self.test_b = self.b.to(self.device), self.test_b.to(self.device)
        if self.args.train_after_joined:
            train_datas = ray_group_call(self.workers, "get_train_data")
            test_datas = ray_group_call(self.workers, "get_test_data")

            self.train_data = np.hstack(train_datas)
            self.test_data = np.hstack(test_datas)

            self.train_dataset = TensorDataset(
                torch.from_numpy(self.train_data), self.b
            )
            self.test_dataset = TensorDataset(
                torch.from_numpy(self.test_data), self.test_b
            )
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=self.args.batch_size, shuffle=False
            )
            self.test_dataloader = DataLoader(
                self.test_dataset, batch_size=self.args.batch_size, shuffle=False
            )
            self.n = self.train_data.shape[1]
            self._model = Linear if self.args.model == "Linear" else MLP
            self.model = self._model(self.n, self.C).double().to(self.device)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )

    def train_per_epoch(self):
        if self.args.train_after_joined:
            return self.joined_train_per_epoch()
        else:
            return self.VFL_train_per_epoch()

    def test(self):
        if self.args.train_after_joined:
            return self.joined_test()
        else:
            return self.VFL_test()

    def joined_train_per_epoch(self):
        self.total_loss = 0
        if self.task == "classification":
            if self.args.metric == "AUC":
                self.train_metric = BinaryAUROC()
            else:
                self.acc = 0
        self.model.train()
        for input, label in self.train_dataloader:
            input, label = input.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input)
            if self.task == "regression":
                label = label.unsqueeze(1)
            loss = self.loss(output, label)
            loss.backward()
            self.optimizer.step()
            self.total_loss += loss.item() * len(label) / self.M
            if self.task == "classification":
                if self.args.metric == "AUC":
                    self.train_metric.update(torch.softmax(output, dim=1)[:, 1], label)
                else:
                    predicted = torch.argmax(output, 1)
                    self.acc += (predicted == label).sum().item() / self.M
        return loss

    def joined_test(self):
        self.test_total_loss = 0
        if self.task == "classification":
            if self.args.metric == "AUC":
                self.test_metric = BinaryAUROC()
            else:
                self.test_acc = 0
        self.model.eval()
        for input, label in self.test_dataloader:
            input, label = input.to(self.device), label.to(self.device)
            with torch.no_grad():
                output = self.model(input)
                if self.task == "regression":
                    label = label.unsqueeze(1)
            loss = self.loss(output, label)
            self.test_total_loss += loss.item() * len(label) / self.test_M
            if self.task == "classification":
                if self.args.metric == "AUC":
                    self.test_metric.update(torch.softmax(output, dim=1)[:, 1], label)
                else:
                    predicted = torch.argmax(output, 1)
                    self.test_acc += (predicted == label).sum().item() / self.test_M
        return self.validate("test")

    def VFL_train_per_epoch(self):
        self.total_loss = 0
        if self.task == "classification":
            if self.args.metric == "AUC":
                self.train_metric = BinaryAUROC()
            else:
                self.acc = 0
        ray_group_call(self.workers, "prepare_dataloader_iter")
        for batch_id in range(self.train_iterations):
            sample_start_idx = batch_id * self.args.batch_size
            sample_end_idx = (batch_id + 1) * self.args.batch_size
            label = self.b[sample_start_idx:sample_end_idx].to(self.device)
            batch_f = torch.from_numpy(self.f[sample_start_idx:sample_end_idx, :]).to(
                self.device
            )
            loss, T_x = self._process_batch(label, batch_f=batch_f)
            loss.backward()
            # Here we need to pass grad to worker
            grad_args = [(t.grad,) for t in T_x]
            ray_group_call_multiargs(self.workers, "receive_embedding_grad", grad_args)
            if self.args.use_bcd:
                ray_group_call(self.workers, "bcd_one_step_backward")
            else:
                ray_group_call(self.workers, "one_step_backward")

    def _process_batch(self, label, mode="train", batch_f=None):
        if self.task == "regression":
            label = label.reshape(-1, 1).double()
        T_x = []
        
        if self.args.opt:
            unique_values_list = []
            inverse_indices_list = []
            for i in range(self.N):
                unique_values, inverse_indices = torch.unique(batch_f[:, i + 1], return_inverse=True)
                unique_values_list.append(unique_values)
                inverse_indices_list.append(inverse_indices)
            prep_args = [(mode, unique_values_list[i]) for i in range(self.N)]
            ray_group_call_multiargs(self.workers, "prepare_batch", prep_args)
            ray_group_call(self.workers, "one_step_forward", mode)
            embeddings = ray_group_call(self.workers, "get_embedding")
            T_x = [
                torch.index_select(torch.as_tensor(embeddings[i]), 0, inverse_indices_list[i])
                for i in range(self.N)
            ]
            set_args = [(inverse_indices_list[i],) for i in range(self.N)]
            ray_group_call_multiargs(self.workers, "set_inverse_indices", set_args)

        else:
            ray_group_call(self.workers, "prepare_batch", mode)
            ray_group_call(self.workers, "one_step_forward", mode)
            T_x = ray_group_call(self.workers, "get_embedding")

        h = torch.sum(torch.stack(T_x), dim=0)
        loss = self.loss(h, label)
        if mode == "train":
            self.total_loss += loss.item() * len(label) / self.M
        else:
            self.test_total_loss += loss.item() * len(label) / self.test_M
        if self.task == "classification":
            if self.args.metric == "AUC":
                if mode == "train":
                    self.train_metric.update(torch.softmax(h, dim=1)[:, 1], label)
                else:
                    self.test_metric.update(torch.softmax(h, dim=1)[:, 1], label)
            else:
                _, predicted = torch.max(h.data, 1)
                if mode == "train":
                    self.acc += (predicted == label).sum().item() / self.M
                else:
                    self.test_acc += (predicted == label).sum().item() / self.test_M
        return loss, T_x

    def validate(self, mode="train"):
        if mode == "train":
            total_loss = self.total_loss
            if self.task == "classification":
                if self.args.metric == "AUC":
                    acc = self.train_metric.compute().item()
                else:
                    acc = self.acc
        else:
            total_loss = self.test_total_loss
            if self.task == "classification":
                if self.args.metric == "AUC":
                    acc = self.test_metric.compute().item()
                else:
                    acc = self.test_acc
        loss = total_loss
        if self.task == "classification":
            return loss, acc
        elif self.task == "regression":
            rmse = math.sqrt(loss)
            return rmse

    def VFL_test(self):
        self.test_total_loss = 0
        if self.task == "classification":
            if self.args.metric == "AUC":
                self.test_metric = BinaryAUROC()
            else:
                self.test_acc = 0
        for batch_id in range(self.test_iterations):
            sample_start_idx = batch_id * self.args.batch_size
            sample_end_idx = (batch_id + 1) * self.args.batch_size
            label = self.test_b[sample_start_idx:sample_end_idx].to(self.device)
            batch_f = torch.from_numpy(
                self.test_f[sample_start_idx:sample_end_idx, :]
            ).to(self.device)
            self._process_batch(label, mode="test", batch_f=batch_f)
        return self.validate("test")
