import numpy as np
import matplotlib.pyplot as plt
import utils


class underlying:
    def __init__(self, S0: float, r: float, sigma: float):
        self.r = r
        self.S0 = S0
        self.sigma = sigma

    def simulate(self, simulation: str, T: float, n_time: int = None, n_space: int = None, n_mc: int = None):
        match simulation:
            case "BS":
                W = utils.brownian_motion(T, n_time, n_mc)
                t = np.linspace(0, T, n_time + 1)
                return utils.BS_process(self.S0, self.r, self.sigma, t, W)

    def plot_paths(self, T: float, n_time: int, n_mc: int):
        paths = self.simulate(simulation="BS", T=T, n_time=n_time, n_mc=n_mc)
        t = np.linspace(0, T, n_time + 1)
        fig = plt.figure(figsize=(12, 6))
        if (n_mc <= 5):
            for i in range(paths.shape[0]):
                plt.plot(t, paths[i, :], label=f"{i + 1} path")
        else:
            for i in range(paths.shape[0]):
                plt.plot(t, paths[i, :])
        plt.xlabel("time")
        plt.ylabel("stock price")
        plt.title(f"Example of stock paths | BS Model | S0 = {self.S0}, r = {self.r}, $\sigma$ = {self.sigma}")
        plt.grid()
        if(n_mc <= 5):
            plt.legend()
        plt.show()
