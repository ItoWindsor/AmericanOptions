import numpy as np
from scipy.sparse import diags
from typing import Callable

def compute_system_matrix(tau : float, h : float, x : np.array, sigma : float, r : float):

    a = (tau * r * x) / (2 * h) - ((sigma ** 2) * (x ** 2) * tau) / (2 * (h ** 2))
    b = ((sigma ** 2) * (x ** 2) * tau) / (h ** 2) + tau * r + 1
    c = -(tau * r * x) / (2 * h) - ((sigma ** 2) * (x ** 2) * tau) / (2 * (h ** 2))

    main_diag = b[1:-1]
    lower_diag = a[2:-1]
    upper_diag = c[1:-2]

    return (diags([main_diag, lower_diag, upper_diag], offsets = [0, -1, 1]).toarray(),a,b,c,)

def init_solution_matrix(payoff : Callable, bound_min : Callable, bound_max : Callable, x : np.array, K : float, t : np.array, T : float, r : float, n_time : int, n_space : int):
    V = np.zeros(shape=(n_space + 1, n_time + 1))

    V[:, -1] = payoff(x, K)  # condition terminale (K-e^{s})^{+}
    V[0, :] = bound_min(t, K, T, r)  # condition en x = s_min
    V[-1, :] = bound_max(t)  # condition en x = s_max

    return(V)
