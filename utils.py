import numpy as np


# d rows and N columns
def brownian_motion(T: float, N: int, d: int):
    if d == 1:
        return np.insert(np.random.normal(scale=np.sqrt(T / N), size=N).cumsum(), 0, 0)
    elif d > 1:
        return np.insert(np.random.normal(scale=np.sqrt(T / N), size=(d, N)).cumsum(axis=1), 0, 0, axis=1)


payoff_call = lambda STOCK_PRICE, K: (STOCK_PRICE - K) * (STOCK_PRICE - K >= 0)
payoff_put = lambda STOCK_PRICE, K: (K - STOCK_PRICE) * (K - STOCK_PRICE >= 0)
BS_process = lambda x0, r, sigma, t, W: x0 * np.exp((r - (sigma ** 2) / 2) * t + sigma * W)
