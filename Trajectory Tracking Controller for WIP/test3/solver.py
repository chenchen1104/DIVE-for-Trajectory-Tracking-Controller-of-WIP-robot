import numpy as np
import random
import torch
import torch.nn as nn

TH_DTYPE = torch.float32

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


class FCSubNet(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        layer_dims = [self.dim] + config.num_hiddens  # layer_dims: [2, 32, 128, 32]
        self.bn_layers = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(self.dim, eps=1e-6, momentum=0.99)])
        self.dense_layers = torch.nn.ModuleList([])
        for i in range(len(layer_dims) - 1):
            self.dense_layers.append(torch.nn.Linear(
                layer_dims[i], layer_dims[i + 1], bias=False))
            self.bn_layers.append(torch.nn.BatchNorm1d(
                layer_dims[i + 1], eps=1e-6, momentum=0.99))

        # output layers
        self.dense_layers.append(torch.nn.Linear(
            layer_dims[-1], self.dim, bias=True))
        self.bn_layers.append(torch.nn.BatchNorm1d(
            self.dim, eps=1e-6, momentum=0.99))

        # initializing batchnorm layers
        for layer in self.bn_layers:
            torch.nn.init.uniform_(layer.weight, 0.1, 0.5)
            torch.nn.init.normal_(layer.bias, 0.0, 0.1)

        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.tanh(x)
        x = self.dense_layers[-1](x)
        return x / self.dim


class FCLSTMSubNet(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        layer_dims = [self.dim, config.lstm_hidden_size] + config.num_hiddens  # layer_dims: [2,20,32,128,32]
        self._layers = torch.nn.ModuleList([])
        self._layers.append(
            torch.nn.LSTM(input_size=self.dim, hidden_size=config.lstm_hidden_size, num_layers=config.lstm_num_layers))
        for i in range(1, len(layer_dims) - 1):
            self._layers.append(torch.nn.Linear(
                layer_dims[i], layer_dims[i + 1], bias=False))

        # output layers
        self._layers.append(torch.nn.Linear(
            layer_dims[-1], self.dim, bias=True))

        self.relu = torch.nn.Tanh()

    def forward(self, x, hidden):
        x = x.unsqueeze(0)
        x, hidden_ = self._layers[0](x, hidden)
        x = x.squeeze(0)
        for i in range(len(self._layers) - 2):
            x = self._layers[i + 1](x)
            x = self.relu(x)
        x = self._layers[-1](x)
        return x / self.dim, hidden_


def my_sig(x):
    return 2.0 / (1 + torch.exp(-x)) - 1


class FeedForwardModel(nn.Module):
    """The fully connected neural network model."""

    def __init__(self, config, fbsde):
        super(FeedForwardModel, self).__init__()
        self._config = config
        self.fbsde = fbsde

        self._dim = fbsde.dim
        self._num_time_interval = fbsde.num_time_interval
        self._total_time = fbsde.total_time

        self.register_parameter('y_init', torch.nn.Parameter(
            torch.rand(1).uniform_(config.y_init_range[0], config.y_init_range[1])))
        self._subnetworkList = nn.ModuleList([FCSubNet(config)])

        if config.lstm == True:
            self._subnetworkList = nn.ModuleList([FCLSTMSubNet(config)])
        else:
            if config.fcsame == True:
                self._subnetworkList = nn.ModuleList([FCSubNet(config)])
            else:
                self._subnetworkList = nn.ModuleList([FCSubNet(config) for _ in range(self._num_time_interval)])

    def x_desired(self, length, t):
        X_bar = torch.zeros([length, self.fbsde.dim, 1])
        X_bar[:, 0, 0] = (self.fbsde.a * t - t ** 2) * np.exp(-self.fbsde.alpha * t)
        X_bar[:, 1, 0] = (self.fbsde.a - 2 * t) * np.exp(-self.fbsde.alpha * t) - self.fbsde.alpha * (
                self.fbsde.a * t - t ** 2) * np.exp(-self.fbsde.alpha * t)
        return X_bar

    def forward(self, dw):
        num_sample = dw.shape[0]

        R = self.fbsde.R

        all_one_vec = torch.ones((num_sample, 1), dtype=TH_DTYPE)
        y = all_one_vec * self.y_init
        y = y.unsqueeze(2)

        error_x = torch.zeros([num_sample, self._dim, 1])

        totalx = []
        totalu = []

        time_stamp = np.arange(0, self.fbsde.num_time_interval) * self.fbsde.delta_t
        hidden = (torch.randn(self._config.lstm_num_layers, num_sample, self._config.lstm_hidden_size),
                  torch.randn(self._config.lstm_num_layers, num_sample, self._config.lstm_hidden_size))

        for t in range(0, self._num_time_interval):
            x_desired = self.x_desired(num_sample, time_stamp[t])
            if t == 0:
                error_x = x_desired
            x_sample = x_desired - error_x
            totalx.append(x_sample)

            if self._config.lstm == True:
                z, hidden = self._subnetworkList[0](x_sample.squeeze(2), hidden)
            else:
                if self._config.fcsame == True:
                    z = self._subnetworkList[0](x_sample.squeeze(2))
                else:
                    z = self._subnetworkList[t](x_sample.squeeze(2))
            z = z.unsqueeze(2)

            gamma = self.fbsde.gamma_(x_sample)
            u = (-1 / R) * torch.bmm(torch.transpose(gamma, 1, 2), z)
            if self._config.constrained == True:
                u = torch.clamp(u, -self._config.umax, self._config.umax)
            totalu.append(u)

            i1 = self.fbsde.delta_t * self.fbsde.h_th(time_stamp[t], x_sample, error_x, z, u)
            i2 = torch.bmm(torch.transpose(z, 1, 2), gamma)
            i3 = self.fbsde.delta_t * torch.bmm(i2, u)
            y = y - i1 + i3

            w = self.fbsde.w(x_sample, time_stamp[t])
            item1 = torch.bmm(self.fbsde.A_(x_sample), error_x) * self.fbsde.delta_t
            item2 = torch.bmm(self.fbsde.G_(x_sample), u) * self.fbsde.delta_t
            error_x = item1 + item2 + w * self.fbsde.delta_t

        yT = self.fbsde.g_th(self._total_time, error_x, u)
        loss = torch.sum(abs((y.squeeze(2) - yT.squeeze(2))))
        return loss, self.y_init, yT, totalx, totalu
