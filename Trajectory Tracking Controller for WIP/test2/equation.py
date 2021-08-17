import numpy as np
import torch

class Equation(object):
    def __init__(self, dim, total_time, delta_t):
        self._dim = dim
        self._total_time = total_time
        self._delta_t = delta_t
        self._num_time_interval = int(self._total_time / delta_t)
        self._sqrt_delta_t = np.sqrt(self._delta_t)
        self._y_init = None

    def sample(self, num_sample):
        """Sample forward SDE."""
        raise NotImplementedError

    def f_th(self, x):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_th(self, t, x, u):
        """Terminal condition of the PDE."""
        raise NotImplementedError

    @property
    def y_init(self):
        return self._y_init

    @property
    def dim(self):
        return self._dim

    @property
    def num_time_interval(self):
        return self._num_time_interval

    @property
    def total_time(self):
        return self._total_time

    @property
    def delta_t(self):
        return self._delta_t


class WIP_LINEAR(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(WIP_LINEAR, self).__init__(dim, total_time, num_time_interval)
        self.mb = 15
        self.mw = 0.42
        self.L = 0.2
        self.R = 0.106
        self.Ib2 = 0.63
        self.Ib3 = 1.12
        self.g = 9.81
        self.d = 0.212
        self.alpha = 0.5
        self.a = 20

        self.D_IPL1 = (self.mb + 3 * self.mw) * (self.mb * self.d ** 2 + self.Ib3) - (self.mb * self.d) ** 2
        self.D_IPL2 = self.Ib2 + (5 * self.L ** 2 + self.R ** 2 / 2) * self.mw

        self.A23L = -(self.mb * self.d ** 2) * self.g / self.D_IPL1
        self.A43L = ((self.mb + 3 * self.mw) * self.mb * self.d * self.g) / self.D_IPL1

        self.B21L = (self.mb * self.d ** 2 + self.Ib3) / (self.D_IPL1 * self.R)
        self.B22L = self.B21L
        self.B41L = (-self.mb * self.d) / (self.D_IPL1 * self.R)
        self.B42L = self.B41L
        self.B61L = self.L / (self.D_IPL2 * self.R)
        self.B62L = -self.B61L

        self.A = torch.tensor([[0, 1, 0, 0], [0, 0, self.A23L, 0], [0, 0, 0, 1], [0, 0, self.A43L, 0]])
        d = torch.tensor([0.1, 0.1, 0.1, 0.1])
        self.sigma = torch.diag(d)
        self.G = torch.tensor([[0], [self.B21L], [0], [self.B41L]])
        self.gamma = torch.mm(self.sigma.inverse(), self.G)
        self.R = 1
        self.Q = torch.tensor([[10000, 0, 0, 0], [0, 0, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0]])

        self.X_bar = torch.zeros((4, 1))
        self.dX_bar = torch.zeros((4, 1))

    def sample(self, num_sample):
        dw = torch.empty(num_sample, self.dim, self.num_time_interval).normal_(std=self._sqrt_delta_t)
        return dw

    def f_th(self, x):
        f1 = x[:, 1]
        f2 = self.A23L * x[:, 2]
        f3 = x[:, 3]
        f4 = self.A43L * x[:, 2]
        f = torch.cat((f1, f2, f3, f4), 1)
        return f

    def w(self, length, t):
        self.X_bar[0] = (self.a * t - t ** 2) * np.exp(-self.alpha * t)
        self.X_bar[1] = (self.a - 2 * t) * np.exp(-self.alpha * t) - self.alpha * (self.a * t - t ** 2) * np.exp(
            -self.alpha * t)
        self.dX_bar[0] = (self.a - 2 * t) * np.exp(-self.alpha * t) - self.alpha * (self.a * t - t ** 2) * np.exp(
            -self.alpha * t)
        self.dX_bar[1] = -2 * np.exp(-self.alpha * t) + self.alpha * (2 * t - self.a) * np.exp(
            -self.alpha * t) - self.alpha * (self.a - 2 * t) * np.exp(-self.alpha * t) + self.alpha ** 2 * (
                                 self.a * t - t ** 2) * np.exp(-self.alpha * t)

        dX_bar = torch.zeros([length, self.dim, 1])
        dX_bar[:, ] = self.dX_bar
        X_bar = torch.zeros([length, self.dim, 1])
        X_bar[:, ] = self.X_bar
        w = torch.bmm(self.A_(length), X_bar) - dX_bar
        return w

    def A_(self, length):
        A_ = torch.zeros([length, self.dim, self.dim])
        A_[:, ] = self.A
        return A_

    def Q_(self, length):
        Q_ = torch.zeros([length, self.dim, self.dim])
        Q_[:, ] = self.Q
        return Q_

    def sigma(self, length):
        sigma_ = torch.zeros([length, self.dim, self.dim])
        sigma_[:, ] = self.sigma
        return sigma_

    def gamma_(self, length):
        gamma_ = torch.zeros([length, self.dim, 1])
        gamma_[:, ] = self.gamma
        return gamma_

    def G_(self, length):
        G_ = torch.zeros([length, self.dim, 1])
        G_[:, ] = self.G
        return G_

    def g_th(self, t, x, u):
        Q = self.Q_(x.shape[0])
        g = torch.bmm(torch.transpose(x, 1, 2), Q)
        g = torch.bmm(g, x)
        g = torch.sum(g, dim=1)
        g = g.unsqueeze(2) + self.R * u ** 2
        return g

    def h_th(self, t, error_x, z, u):
        gamma = self.gamma_(z.shape[0])
        temp = torch.bmm(torch.transpose(z, 1, 2), gamma) * (1 / self.R)
        temp1 = torch.bmm(temp, torch.transpose(gamma, 1, 2))
        h = self.g_th(t, error_x, u) - torch.bmm(temp1, z) / 2
        return h


def get_equation(name, dim, total_time, num_time_interval):
    try:
        return globals()[name](dim, total_time, num_time_interval)
    except KeyError:
        raise KeyError("Equation for the required problem not found.")
