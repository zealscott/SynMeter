import math

import numpy as np
from scipy.optimize import fsolve


# use below functions in synthesizer.py to determine noise type and corresponding noise parameters

def lap_comp(epsilon, delta, sensitivity, k):
    return epsilon * 1.0 / k / sensitivity


def lap_adv_comp(epsilon, delta, sensitivity, k):
    def func(x_0):
        eps_0 = x_0[0]
        return math.sqrt(2 * k * math.log(1 / delta)) * eps_0 + k * (math.exp(eps_0) - 1) * eps_0 - epsilon

    result = fsolve(func, np.array([0.0]))

    return result[0] / sensitivity


def gauss_adv_comp(epsilon, delta, sensitivity, k):
    def gauss(delta_0):
        dlt = delta - delta_0 * k

        def eps_func(x_0):
            eps_0 = x_0[0]
            return math.sqrt(2 * k * math.log(1 / dlt)) * eps_0 + k * (math.exp(eps_0) - 1) * eps_0 - epsilon

        epsilon_0 = fsolve(eps_func, np.array([0.0]))[0]
        sigma = sensitivity * np.sqrt(2 * math.log(1.25 / delta_0)) / epsilon_0
        return sigma

    l, h = 1e-30, delta * 1.0 / k / 1.1
    min_delta_0 = my_minimize(gauss, l, h)
    return gauss(min_delta_0)


def my_minimize(func, l, h):
    vfunc = np.vectorize(func)
    cur_l, cur_h = l, h
    n = 20000
    for i in range(10):
        xs = np.linspace(cur_l, cur_h, n)
        vs = vfunc(xs)
        vs_index = np.argsort(vs)
        cur_l_index, cur_h_index = vs_index[0], vs_index[1]
        cur_l, cur_h = xs[cur_l_index], xs[cur_h_index]

    return (cur_l + cur_h) / 2


def gauss_renyi(epsilon, delta, sensitivity, k):
    def renyi(low):
        epsilon0 = max(1e-20, epsilon - np.log(1.0 / delta) * 1.0 / (low - 1))
        sigma = np.sqrt(k * low * sensitivity ** 2 * 1.0 / 2 / epsilon0)
        return sigma

    l, h = 1.00001, 100000
    min_low = my_minimize(renyi, l, h)
    min_sigma = renyi(min_low)

    return min_sigma


def gauss_zcdp(epsilon, delta, sensitivity, k):
    tmp_var = 2 * k * sensitivity ** 2 * math.log(1 / delta)

    sigma = (math.sqrt(tmp_var) + math.sqrt(tmp_var + 2 * k * sensitivity ** 2 * epsilon)) / (2 * epsilon)

    return sigma


# zcdp and zcdp2 and rdp perform the same
def gauss_zcdp2(epsilon, delta, sensitivity, k):
    my_log = math.log(1 / delta)

    sigma = sensitivity * math.sqrt(k / 2) / (math.sqrt(epsilon + my_log) - math.sqrt(my_log))

    return sigma


def lap_zcdp_comp(epsilon, delta, sensitivity, k):
    return math.sqrt(2.0 * (math.sqrt(k) * sensitivity / epsilon) ** 2)


def get_noise(eps, delta, sensitivity, num_composition):
    lap_param = lap_comp(eps, delta, sensitivity, num_composition)
    lap_naive_var = 2 * (1.0 / lap_param ** 2)

    gauss_param = gauss_zcdp(eps, delta, sensitivity, num_composition)
    gauss_var_zcdp = gauss_param ** 2
    if lap_naive_var < gauss_var_zcdp:
        return 'lap', 1 / lap_param
    else:
        return 'gauss', gauss_param


