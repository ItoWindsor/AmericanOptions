import numpy as np
from typing import Callable
from scipy.optimize import curve_fit
from scipy.linalg import lu_factor, lu_solve

import Underlying
import FiniteDifference as fd
class option:

    def __init__(self, S0: float, r: float, sigma: float, T: float, K : float, payoff : Callable, bound_min : Callable = None, bound_max : Callable = None, x_min : float = 0.00001, x_max : float = None, kind_option : str = "european", simulation : str = 'FD'):
        self.underlying = Underlying.underlying(S0 = S0, r = r, sigma = sigma)
        self.T = T
        self.K = K
        self.payoff = payoff
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.x_min = x_min
        self.x_max = x_max * (x_max is not None) + 4 * self.K * (x_max is None)
        x_max = 4 * K
        self.kind_option = kind_option
        self.simulation = simulation

    def compute_price(self,simulation : str = None, n_time : int = None, n_space : int = None, n_mc : int = None):
        if simulation is None:
            simulation = self.simulation

        match simulation:
            case "FD":
                t = np.linspace(0, self.T, n_time + 1)
                x = np.linspace(self.x_min, self.x_max, n_space + 1)
                tau = self.T / n_time
                h = (self.x_max - self.x_min) / n_time

                A,a,b,c= fd.compute_system_matrix(tau, h, x, self.underlying.sigma, self.underlying.r)
                lu, piv = lu_factor(A)
                V = fd.init_solution_matrix(self.payoff, self.bound_min, self.bound_max, x, self.K, t, self.T, self.underlying.r, n_time, n_space)
                PAYOFF = self.payoff(x, self.K)

                for j in range(n_time - 1, -1, -1):
                    B = np.zeros(n_space - 1)
                    B[0] = a[1] * V[0, j]
                    B[-1] = c[-1] * V[-1, j]
                    V_temp = lu_solve((lu, piv), V[1:-1, j + 1] - B)  # solution of the PDE
                    match self.kind_option:
                        case "european":
                            V[1:-1, j] = V_temp
                        case "american":
                            V[1:-1, j] = np.maximum(V_temp, PAYOFF[1:-1])
                            V[0, j] = np.maximum(V[0, j], PAYOFF[0])
                            V[-1, j] = np.maximum(V[-1, j], PAYOFF[-1])
                return t,V

            case "MonteCarlo":
                return -1
            case "LS":
                # Longstaff-Schwartz regression basis => HARDCODED SO FAR
                basis1 = lambda x: 1
                basis2 = lambda x: x
                basis3 = lambda x: x ** 2

                f = lambda x, alpha1, alpha2, alpha3: alpha1 * basis1(x) + alpha2 * basis2(x) + alpha3 * basis3(x)  # Longstaff-Schwartz regression function

                t = np.linspace(0, self.T, n_time + 1)
                # m = len(t[0])
                D = np.exp(-self.underlying.r * (self.T / n_time))

                price_arr = np.zeros(10)  ## hardcoded so far
                for k in range(10):  ## hardcoded so far
                    S = self.underlying.simulate(simulation="BS", T=self.T, n_time=n_time,
                                                 n_mc=n_mc)  # St[i] gives the i-th trajectory therefore St[i,k] = S_{t_k}^{i}

                    V = np.zeros(S.shape)
                    V[:, -1] = self.payoff(S[:, -1], self.K)  # the terminal value can only be the payoff of the option
                    alphas = []
                    for i in range(n_time - 1, 0, -1):  # loop backward in time
                        alpha, _ = curve_fit(f, xdata=S[:, i], ydata=V[:, i + 1])  # y_{data} = f(x_{data}) + \varepsilon
                        alphas.append(alpha)

                        for j in range(n_mc - 1, -1, -1):  # loop in sample paths
                            if self.payoff(S[j, i], self.K) >= D * f(S[j, i], alpha[0], alpha[1], alpha[2]) :
                                V[j, i] = self.payoff(S[j, i], self.K)
                            else:
                                V[j, i] = D * V[j, i + 1]
                    price_arr[k] = max(self.payoff(self.underlying.S0, self.K), D * V[:, 1].mean())

                price_MC = price_arr.mean()
                std = np.std(price_arr)
                IC = [price_MC - 1.96 * std / np.sqrt(10), price_MC + 1.96 * std / np.sqrt(10)]
                return price_MC, IC





