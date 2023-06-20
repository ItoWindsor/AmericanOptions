import Underlying
import Option
import utils

S0 = 100
K = 100
r = 0.06
sigma = 0.4

T = 0.5

n_time = int(300 * T)
n_mc = 5000

#asset = Underlying.underlying(S0, r, sigma)

#asset.plot_paths(T, n_time, n_mc)

PutOption = Option.option(S0, r, sigma, T, K, utils.payoff_put, "LS")

price,IC = PutOption.compute_price(n_time=n_time, n_mc=n_mc)
print(f"price : {price}")
print(f"IC : {IC}")
