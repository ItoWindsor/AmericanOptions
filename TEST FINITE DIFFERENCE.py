import numpy as np
import matplotlib.pyplot as plt
import Option

# financial data
r_input = 0.06
high_rate = 1
sigma = 0.4
S0 = 100
K = 100
T = 1

# numerical data
n_time = 200
n_space = 400


x_min = 0.000001
x_max = 4*K

tau = T/n_time
h = (x_max-x_min)/n_space

payoff_put = lambda x,K : (K-x)*(K-x >=0)

bound_min = lambda t,K,T,r : np.exp(-r*(T-t))*K
bound_max = lambda t : 0

put_option_european = Option.option(S0 = 0, r = r_input, sigma = sigma, T = T, K = K, payoff = payoff_put,
                           bound_min = bound_min, bound_max = bound_max, x_min = x_min, x_max = x_max,
                           kind_option= "european", simulation= "FD")

put_option_american = Option.option(S0 = 0, r = r_input, sigma = sigma, T = T, K = K, payoff = payoff_put,
                           bound_min = bound_min, bound_max = bound_max, x_min = x_min, x_max = x_max,
                           kind_option= "american", simulation= "FD")

t,V_euro = put_option_european.compute_price(n_time = n_time, n_space = n_space)
t,V_american = put_option_american.compute_price(n_time = n_time, n_space = n_space)

x = np.linspace(x_min,x_max, n_space + 1)
PAYOFF = payoff_put(x,K)

val_arr = [0,0.25,0.5,0.75]
time_indx = []
for value in val_arr:
    diff = [abs(val - value) for val in t]
    time_indx.append(diff.index(min(diff)))

fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(121);ax2 = fig.add_subplot(122)

for i in time_indx:
    ax1.plot(x, V_euro[:, i], label="t = {}".format(t[i]))
    ax2.plot(x, V_american[:, i], label="t = {}".format(t[i]))

ax1.plot(x, PAYOFF, label="payoff", color='black')
ax2.plot(x, PAYOFF, label="payoff", color='black')


fig.suptitle("Comparison Evolution of the price of a PUT option through time | EU vs US | PDE | T = {}, K = {}, r = {}, $\sigma$ = {} | $\\tau$ = {}, h = {}".format(T, K, r_input, sigma, tau, h), fontsize=16)
fig.subplots_adjust(top=0.85)

ax1.grid();
ax1.legend();
ax1.set_xlabel("asset price");
ax1.set_ylabel("value")
ax1.set_title("european")

ax2.grid();
ax2.legend();
ax2.set_xlabel("asset price");
ax2.set_ylabel("value")
ax2.set_title("american")

plt.show()

