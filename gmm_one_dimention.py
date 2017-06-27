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

def gaussian(mu,sigma):
    def f(x):
        return np.exp(-0.5 * (x - mu) ** 2 / sigma) / np.sqrt(2 * np.pi * sigma)
    return f

def estimate_posterior_likelihood(X,pi, gf):
    l = np.zeros((X.size, pi.size))
    for (i, x) in enumerate(X):
        l[i,:] = gf(x)
    return pi * l * np.vectorize(lambda y: 1/y)(l.sum(axis = 1).reshape(-1,1))

def estimate_gmm_parameter(X, gamma):
    N = gamma.sum(axis=0)
    mu = (gamma * X.reshape((-1,1))).sum(axis=0)/N
    sigma = (gamma * (X.reshape(-1,1) - mu) ** 2).sum(axis=0)/N
    pi = N/X.size
    return (mu,sigma, pi)
def calc_Q(X,mu, sigma, pi ,gamma):
    Q = (gamma * (np.log(pi * (2 * np.pi * sigma) ** (-0.5)))).sum()
    for (i, x) in enumerate(X):
        Q += (gamma[i,:] * (-0.5 *  (x -mu)** 2 / sigma)).sum()
    return Q
def main():
    class_num = 2
    data_num = 1000 * class_num
    X, mu_star, sigma_star = generate_data(class_num,data_num)

    epsilon = 0.000001
    pi = np.random.rand(class_num)
    mu = np.random.randn(class_num)
    sigma = np.abs(np.random.randn(class_num))
    Q = -sys.float_info.max
    delta = None

    while delta == None or delta >= epsilon:
        gf = gaussian(mu, sigma)
        gamma = estimate_posterior_likelihood(X, pi, gf)
        mu, sigma, pi = estimate_gmm_parameter(X, gamma)

        Q_new = calc_Q(X, mu, sigma, pi, gamma)
        delta = Q_new - Q
        Q = Q_new

    print('mu* : {0} sigma* : {1}'.format(np.sort(np.around(mu_star,3)),np.sort(np.around(sigma_star,3))))
    print('mu  : {0} sigma  : {1}'.format(np.sort(np.around(mu,3)),np.sort(np.around(sigma,3))))

    n, bins,_ = pyplot.hist(X, 50, normed = 1, alpha = 0.3)
    seq = np.arange(-15,15,0.02)
    for i in range(class_num):
        pyplot.plot(seq, gaussian(mu[i],sigma[i])(seq), linewidth= 2.0)
    pyplot.show()



if __name__ == '__main__':
    main()
