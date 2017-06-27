import numpy as np
from matplotlib import pyplot
import sys
import argparse

def arg():
    pass

def generate_data(K,N):
    X, mu_star, sigma_star = [], [],[] 
    for i in range(K):
        loc = (np.random.rand() - 0.5) * 10.0
        scale = np.random.rand() + 3.0
        X = np.append(X, np.random.normal(loc = loc, scale = scale, size = int(N/K)))
        mu_star = np.append(mu_star, loc)
        sigma_star = np.append(sigma_star, scale)
    return (X, mu_star, sigma_star)

def main():
    class_num = 2
    data_num = 1000 * class_num
    X, mu_star, sigma_star = generate_data(class_num,data_num)

    epsilon = 0.000001


if __name__ == '__main__':
    main()
