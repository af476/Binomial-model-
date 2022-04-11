# load packages
import numpy as np
from numpy import *

import time
start_time = time.time()

############ binomial option function
def binomial_option(spot, strike, steps, v, output):
    """
    @:param
    :param spot: int or float, spot price
    :param strike: int or flot, strike price
    :param steps: int, number of periods
    :param v: int or float, upward or downward changes per period
    :param output: int, [0: price, 1: payoff, 2: option value, 3: option delta]
    :return: ndarray
    An array object of price, payoff, option value and delta as specified by the output parameter
    """

    # define parameters
    u = 1 + v  # u is up factor
    v = 1 - v  # v is down factor
    p = 0.5  # p here is risk neutral probability (p') - for ease of use


    # initialize arrays
    px = zeros((steps + 1, steps + 1))  # price path
    cp = zeros((steps + 1, steps + 1))  # call intrinsic payoff
    V = zeros((steps + 1, steps + 1))  # option value
    d = zeros((steps + 1, steps + 1))  # delta value

    # binomial loop
    for j in range(steps + 1):
        for i in range(j + 1):
            px[i, j] = spot * power(v, i) * power(u, j - i)
            cp[i, j] = maximum(px[i, j] - strike, 0)

    for j in range(steps + 1, 0, -1):
        for i in range(j):
            if (j == steps + 1):
                V[i, j - 1] = cp[i, j - 1]  # terminal payoff
                d[i, j - 1] = 0  # terminal delta
            else:
                V[i, j - 1] = p * V[i, j] + (1 - p) * V[i + 1, j]
                d[i, j - 1] = (V[i, j] - V[i + 1, j]) / (px[i, j] - px[i + 1, j])

    results = around(px, 2), around(cp, 2), around(V, 2), around(d, 4)

    return results[output]

############## calibration function
def calibration(V, K, N):
    """
    @:param
    :param V: value of a European call option
    :param initial: initial values
    :param v_cal: calibrated up factor
    :param step: calibrate steps
    :return:
    return line 1: calibrated value for v
    return line 2: matched location for that Value in a binomial tree
    """
    lower_bound = 1e-2
    upper_bound = 1
    step = 0.01
    temp = []
    for i in np.arange(lower_bound, upper_bound, step):
        temp = np.argwhere(binomial_option(1, K, N, i, 2) == V)
        if temp.size == 0:
            pass
        else:
            print("calibrated v equals ", i)
            print(temp)

############## AM option function
def binomial_option_AM(spot, strike, steps, v, output, type):
    """
    @:param
    :param spot: int or float, spot price
    :param strike: int or flot, strike price
    :param steps: int, number of periods
    :param v: int or float, upward or downward changes per period
    :param output: int, [0: price, 1: payoff, 2: option value]
    :type: int, [1: call, -1: put]
    :Returns
    first element: initial price/ payoff/ value for the AM option
    second element: when is the best execution step
    """
    # define parameters
    u = 1 + v  # u is up factor
    d = 1 - v  # v is down factor
    p = 0.5  # p here is risk neutral probability (p') - for ease of use

    # initialize arrays
    px = zeros((steps + 1, steps + 1))  # price path
    cp = zeros((steps + 1, steps + 1))  # call/ put intrinsic payoff
    V = zeros((steps + 1, steps + 1))  # option value

    # binomial loop
    for j in range(steps + 1):
        for i in range(j + 1):
            px[i, j] = spot * power(d, i) * power(u, j - i)
            cp[i, j] = maximum(type * (px[i, j] - strike), 0)

    flag = 0
    list = []

    for j in range(steps + 1, 0, -1):
        for i in range(j):
            if (j == steps + 1):
                V[i, j - 1] = cp[i, j - 1]  # terminal payoff
            else:
                V[i, j - 1] = p * V[i, j] + (1 - p) * V[i + 1, j]
                if cp[i, j - 1] - V[i, j-1] > 1e-10:
                    flag += 1
                    list.append(j)

    when = steps
    if(flag): when = list[-1]

    results = around(px, 2), around(cp, 2), around(V, 2), around(d, 4)

    return (results[output][0, 0], when)

def max_S(spot, strike, steps, v, output):
    return binomial_option(spot, strike, steps, v, output)[0, steps]

if __name__ == '__main__':
    ########## Q1
    print("#" * 100)
    print("Q1: Implement a function which, "
          "given v and the strike K of a European call option on the asset S, "
          "expiring after N periods, returns its value V ")
    print("#" * 100)
    print("function: value = binomial_option(1, K, N, v, output=2)")
    print("let's try K = 0.8, N = 4, v = 0.35:")
    print(binomial_option(1, 0.8, 5, 0.35, 2))
    print("#" * 100)
    ########## Q2
    print("Q2: Implement a function which, "
          "given the strike K and value V of a European call option on the asset S, "
          "expiring after N periods, calibrates v to match this price.")
    print("#" * 100)
    print("let's try K = 0.8, N = 4, V = 1.66:")
    print(calibration(1.66, 0.8, 4))
    print("DISCUSSION:This part is a little ambiguous, I could either try this iteration method, "
          "Or I could try estimating the volatility (sigma), "
          "apply the BS model, then estimate the u and d")
    print("#" * 100)
    ########## Q3
    print("Q3:Implement a function which, "
          "given v and the strike K of an American call option on the asset S, "
          "expiring after N periods, returns its value.")
    print("#" * 100)
    print("DISCUSSION: It's never optimal to early exercise AM call, "
          "we could discuss it this Thursday. "
          "value for an AM call = EU call."
          "while it different for AM put, I demonstrate both value for AM call and put at here")
    print(binomial_option_AM(1, 0.8, 5, 0.35, 2, 1))
    print(binomial_option_AM(1, 1.1, 5, 0.35, 2, -1))
    print("#" * 100)
    print("Q4: Implement a function which, given v, returns the expectation of max S.")
    print("#" * 100)
    print("let's try K = 0.8, N = 4, v = 0.35:")
    print(max_S(1, 0.8, 5, 0.35, 0))
    print("Double check with the binomial tree (Price):")
    print(binomial_option(1, 0.8, 5, 0.35, 0))
    print("#" * 100)
    print("--- Total run time: %s seconds ---" % (time.time() - start_time))

