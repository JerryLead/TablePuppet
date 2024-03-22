import numpy as np
import torch
import torch.optim as optim
from torcheval.metrics import BinaryAUROC
from loguru import logger
import server.base


class Server(server.base.Server):
    def init(self):
        self.loss = (
            torch.nn.CrossEntropyLoss(reduction="none")
            if self.task == "classification"
            else torch.nn.MSELoss(reduction="none")
        )
        if self.args.simulate_VFL_ADMM:
            self.G = self.aligned_G.copy()
            self.test_G = self.aligned_test_G.copy()
            for i, worker in enumerate(self.workers):
                self.G[i], self.test_G[i] = worker.align(
                    self.G[i],
                    self.test_G[i],
                    train_idx=self.f[:, i + 1],
                    test_idx=self.test_f[:, i + 1],
                )
                worker.init()
        else:
            self.G = self.original_G.copy()
            self.test_G = self.original_test_G.copy()
            for i, worker in enumerate(self.workers):
                self.G[i], self.test_G[i] = worker.align(self.G[i], self.test_G[i])
                worker.init()
        if self.task == "classification":
            self.z = torch.zeros(
                (self.M, self.C), requires_grad=True, device=self.device
            )
            self.z_optimizer = optim.Adam([self.z], lr=self.args.z_update_lr)
        self.y = np.zeros((self.M, self.C))
        self.h = np.zeros((self.M, self.C))
        self.A_x = [np.zeros((self.M, self.C)) for _ in range(self.N)]
        self.s = [np.zeros((self.M, self.C)) for _ in range(self.N)]
        self.Y = [np.zeros((len(self.G[i]), self.C)) for i in range(self.N)]
        self.G_cnt = [np.zeros(len(self.G[i]), dtype=int) for i in range(self.N)]
        for i in range(self.N):
            for j in range(len(self.G_cnt[i])):
                self.G_cnt[i][j] = len(self.G[i][j])

    def train_per_epoch(self):
        rho = self.args.rho
        # Step 2: ADMM z-update
        logger.info("Begin z-update")
        z = self._update_z(rho)
        # Step 3: ADMM y-update
        self.y = self.y + rho * (self.h - z)
        # Step 4: ADMM x-update
        for i in range(self.N):
            self.s[i] = self.h - self.A_x[i] - z
        for i in range(self.N):
            for j in range(len(self.Y[i])):
                self.Y[i][j] = self.y[self.G[i][j]].sum(axis=0) + rho * self.s[i][
                    self.G[i][j]
                ].sum(axis=0)
        logger.info("Begin x-update")
        for i in range(self.N):
            self.workers[i].update_organization_model(self.Y[i], rho)
        T_x = [self.workers[i].logits() for i in range(self.N)]
        self.A_x = [self._map_T_x_to_A_x(T_x[i], self.G[i]) for i in range(self.N)]
        self.h = np.sum(self.A_x, axis=0)

    def _map_T_x_to_A_x(self, T_x, G, mode="train"):
        if mode == "train":
            A_x = np.zeros((self.M, self.C))
        elif mode == "test":
            A_x = np.zeros((self.test_M, self.C))
        cnt = 0
        for i in range(len(T_x)):
            A_x[G[i]] = T_x[i]
            cnt += len(G[i])
        if cnt != len(A_x):
            raise Exception("Expect sum of len(G[i]) to be equal to len(A_x)")
        return A_x

    def _update_z(self, rho):
        if self.task == "classification":
            return self._classification_update_z(rho)
        elif self.task == "regression":
            return self._regression_update_z(rho)

    def _classification_update_z(self, rho):
        b, y, h = (
            torch.from_numpy(self.b).to(self.device),
            torch.from_numpy(self.y).to(self.device),
            torch.from_numpy(self.h).to(self.device),
        )
        for _ in range(self.args.z_update_epoch):
            self.z_optimizer.zero_grad()
            loss = (
                self.loss(self.z, b)
                - torch.sum(y * self.z, dim=1)
                + rho / 2 * torch.linalg.vector_norm(h - self.z, dim=1) ** 2
            )
            loss.backward(torch.ones(loss.shape[0], device=self.device))
            self.z_optimizer.step()
        return self.z.detach().cpu().numpy()

    def _regression_update_z(self, rho):
        return (self.b.reshape(-1, 1) + self.y + rho * self.h) / (1 + rho)

    def validate(self, mode="train"):
        if mode == "train":
            h, b = self.h, self.b
        else:
            h, b = self.test_h, self.test_b
        if self.task == "classification":
            return self._validate_classification(h, b)
        elif self.task == "regression":
            return self._validate_regression(h, b)

    def _validate_classification(self, h, labels):
        avg_loss = 0.0
        total_correct = 0
        M = len(h)
        if self.args.metric == "AUC":
            metric = BinaryAUROC()
        for i in range(0, M, self.args.local_batch_size):
            batch_h = h[i : i + self.args.local_batch_size]
            batch_labels = labels[i : i + self.args.local_batch_size]
            loss = self.loss(torch.from_numpy(batch_h), torch.from_numpy(batch_labels))
            loss = torch.sum(loss).item() / M
            avg_loss += loss
            if self.args.metric == "AUC":
                prob = torch.softmax(torch.from_numpy(batch_h), dim=1)[:, 1]
                metric.update(prob, torch.from_numpy(batch_labels))
            else:
                predictions = np.argmax(batch_h, axis=1)
                correct = (predictions == batch_labels).sum()
                total_correct += correct
        if self.args.metric == "AUC":
            acc = metric.compute().item()
        else:
            acc = total_correct / len(labels)
        return avg_loss, acc

    def _validate_regression(self, h, labels):
        labels = labels[:, None]
        mse_loss = 0.0
        M = len(h)
        for i in range(0, M, self.args.local_batch_size):
            batch_h = h[i : i + self.args.local_batch_size]
            batch_labels = labels[i : i + self.args.local_batch_size]
            loss = np.sum((batch_h - batch_labels) ** 2)
            mse_loss += loss / M
        rmse_loss = np.sqrt(mse_loss)
        return rmse_loss

    def test(self):
        self.test_T_x = [self.workers[i].logits("test") for i in range(self.N)]
        self.test_A_x = [
            self._map_T_x_to_A_x(self.test_T_x[i], self.test_G[i], mode="test")
            for i in range(self.N)
        ]
        self.test_h = np.sum(self.test_A_x, axis=0)
        return self.validate("test")
