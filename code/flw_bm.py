'''
A collection of benchmark optimisation problems to be tested with the
FLW algorithm. Each benchmark is implemented as numpy vectorised function.

Copyright (c) 2023 Sergio Rojas-Galeano and Angie Blanco-Cañon

Version: 1.0 (April/2023)
License: GPLv3 (see https://www.gnu.org/licenses/gpl-3.0.txt)
'''

import numpy as np

######################################
## Real-valued problems (v1.0, SRG) ##
######################################

def sphere(X):
    """ Landscape propierties: Separable, Scalable, Differentiable, Multimodal, Continuous.
    		Domain: [-5.12, 5.12].
    		Dimensions: d >= 2.
    		Optimal function value: 0.
    		Optimal coordinates: (0, ..., 0).
    		Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    return np.sum(X**2, axis=1)

def rosenbrock(X):
    """ Landscape propierties: Non-Separable, Scalable, Differentiable, Unimodal, Continuous.
            Domain: [-5.12, 5.12].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: (1, ..., 1).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    d = X.shape[1]      # Search space dimensionality
    return np.sum([(1 - X[:,i])**2 + 100 * (X[:,i+1] - (X[:,i]**2))**2 for i in np.arange(0, d, 2)], axis=0)
    # return (1 - X[:,0])**2 + 100 * (X[:,1] - (X[:,0]**2))**2

def rastrigin(X):
    """ Landscape propierties: Separable, Differentiable, Multimodal, Continuous.
            Domain: [-5.12, 5.12].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: (0, ..., 0).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    d = X.shape[1]      # Search space dimensionality
    return np.sum((X**2 - 10*np.cos(2*np.pi*X)), axis=1) + 10*d

def rastrigin_bipolar(X):
    """ Landscape propierties: Separable, Differentiable, Multimodal, Continuous.
            Domain: [-5.12, 5.12].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: (-1, 1, -1, 1, ...).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    d = X.shape[1]      # Search space dimensionality
    return np.sum(((X-(2*(np.array(range(0, d)) % 2)-1))**2
            - 10*np.cos(2*np.pi*(X-(2*(np.array(range(0, d)) % 2)-1)))), axis=1) + 10*d

def rastrigin_offset(X):
    """ Landscape propierties: Separable, Differentiable, Multimodal, Continuous.
            Domain: [-5.12, 5.12].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: (1.123, ..., 1.123).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    d = X.shape[1]      # Search space dimensionality
    return np.sum(((X-1.123)**2 - 10*np.cos(2*np.pi*(X-1.123))), axis=1) + 10*d

def himmelblau(X):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-5.12,5.12].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (3,2),(−2.8051,3.2831),(−3.7793,−3.2831),(3.5844,−1.8481).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    return (X[:, 0]**2 + X[:, 1] - 11)**2 + (X[:, 0] + X[:, 1]**2 - 7)**2

def eggholder(X):
    """ Landscape propierties: Non-Separable, Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-512, 512].
            Dimensions: d = 2.
            Optimal function value: −959.640662.
            Optimal coordinates: (512, 404.2319).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    Z = X[:, 1]+47
    return (-Z * np.sin(np.sqrt((np.abs(X[:, 0]/2 + Z)))) \
            -X[:, 0] * np.sin(np.sqrt((np.abs(X[:, 0] - Z))))) #+ 959.640662720851

#################################################
## Additional real-valued problems (v3.0, ABC) ##
#################################################

def ackley(x):
    """ Landscape propierties: Separable, Non-Scalable, Differentiable, Unimodal, Continuous.
            Domain: [-32,32].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: (0, ..., 0).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    d = x.shape[1]     # Search space dimensionality
    return (-20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum(axis=1)))
            - np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum(axis=1))
            + 20.0
            + np.exp(1))

def beale(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Unimodal, Continuous.
            Domain: [-4.5, 4.5].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (3, 0.5).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return ((1.5 - x[:, 0] + x[:, 0] * x[:, 1]) ** 2.0
            + (2.25 - x[:, 0] + x[:, 0] * x[:, 1] ** 2.0) ** 2.0
            + (2.625 - x[:, 0] + x[:, 0] * x[:, 1] ** 3.0) ** 2.0)
			
def booth(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Unimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (1, 3).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return (x[:, 0] + 2 * x[:, 1] - 7) ** 2.0 + (2 * x[:, 0] + x[:, 1] - 5) ** 2.0

def crossintray(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d = 2.
            Optimal function value: −2.06261218.
            Optimal coordinates: (±1.3494067,±1.3494067).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return -0.0001 * np.power(
        np.abs(
            np.sin(x[:, 0])
            * np.sin(x[:, 1])
            * np.exp(np.abs(100 - (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / np.pi)))
        )
        + 1,
        0.1,)
 
def easom(x):
    """ Landscape propierties: Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-100, 100].
            Dimensions: d = 2.
            Optimal function value: -1.
            Optimal coordinates: (π, π).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return (-1
            * np.cos(x[:, 0])
            * np.cos(x[:, 1])
            * np.exp(-1 * ((x[:, 0] - np.pi) ** 2 + (x[:, 1] - np.pi) ** 2)))

def goldstein(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-2, 2].
            Dimensions: d = 2.
            Optimal function value: 3.
            Optimal coordinates: (0, -1).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return (1 + (x[:, 0] + x[:, 1] + 1) ** 2.0 * (19 - 14 * x[:, 0] + 3 * x[:, 0] ** 2.0
                                                  - 14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1] ** 2.0)) \
        * (30 + (2 * x[:, 0] - 3 * x[:, 1]) ** 2.0 * (18 - 32 * x[:, 0] + 12 * x[:, 0] ** 2.0 + 48 * x[:, 1]
                                                      - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1] ** 2.0))


def holdertable(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d = 2.
            Optimal function value: −19.2085.
            Optimal coordinates: (±8.05502, ±9.66459).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    return -np.abs(
        np.sin(x[:, 0])
        * np.cos(x[:, 1])
        * np.exp(np.abs(1 - np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) / np.pi)))

def levy(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Unimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: (1, ..., 1).
            Arguments:
            X -- A population of d-dimensional vector coordinates.
    """
    mask = np.full(x.shape, False)
    mask[:, -1] = True
    masked_x = np.ma.array(x, mask=mask)
    w_ = 1 + (x - 1) / 4
    masked_w_ = np.ma.array(w_, mask=mask)
    d_ = x.shape[1] - 1
    return (np.sin(np.pi * w_[:, 0]) ** 2.0
            + ((masked_x - 1) ** 2.0).sum(axis=1)
            * (1 + 10 * np.sin(np.pi * masked_w_.sum(axis=1) + 1) ** 2.0)
            + (w_[:, d_] - 1) ** 2.0 * (1 + np.sin(2 * np.pi * w_[:, d_]) ** 2.0))

def matyas(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Unimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (0, 0).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return 0.26 * (x[:, 0] ** 2.0 + x[:, 1] ** 2.0) - 0.48 * x[:, 0] * x[:, 1]

def schaffer2(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-100, 100].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (0, 0).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return 0.5 + ((np.sin(x[:, 0] ** 2.0 - x[:, 1] ** 2.0) ** 2.0 - 0.5)
               / ((1 + 0.001 * (x[:, 0] ** 2.0 + x[:, 1] ** 2.0)) ** 2.0))

def threehump(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-5, 5].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (0, 0).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return 2 * x[:, 0] ** 2 - 1.05 * (x[:, 0] ** 4) + (x[:, 0] ** 6) / 6 + x[:, 0] * x[:, 1] + x[:, 1] ** 2

def bohachevsky1(x):
    """ Landscape propierties: Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-100, 100].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (0, 0).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return x[:, 0] ** 2.0 + 2.0 * x[:, 1] ** 2.0 - 0.3 * np.cos(3 * np.pi * x[:, 0]) \
           - 0.4 * np.cos(4 * np.pi * x[:, 1]) + 0.7

def zakharov(x):
    """ Landscape propierties: Non-Separable, Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-5, 10].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: (0, ..., 0).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    d = x.shape[1]		# Search space dimensionality
    return np.sum(x ** 2, axis=1) + (0.5 * np.sum(np.arange(1, d+1) * x, axis=1)) ** 2 \
           + (0.5 * np.sum(np.arange(1, d+1) * x, axis=1)) ** 4
		   
def dixonprice(x):
    """ Landscape propierties: Non-Separable, Scalable, Differentiable, Unimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d >= 2.
            Optimal function value: 0.
            Optimal coordinates: x_i = (2^(2^i -2) / 2^i).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    d = x.shape[1]		# Search space dimensionality
    return (x[:, 0] - 1) ** 2 + np.sum(np.arange(2, d+1) * (2.0 * x[:, 1:] ** 2.0 - x[:, -1]) ** 2.0, axis=1)

def michalewicz(x):
    """ Landscape propierties: Multimodal.
            Domain: [0, π].
            Dimensions: d >= 2.
            Optimal function value: When d = 2: −1.8013; When d = 5: −4.687658, When d = 10: −9.66015.
            Optimal coordinates: When d = 2: (2.20, 1.57).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    d = x.shape[1]		# Search space dimensionality
    return -np.sum(np.sin(x) * (np.sin( np.arange(1, d) * x ** 2 / np.pi)) ** 20, axis=1)

def mishra3(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d = 2.
            Optimal function value: −0.18467.
            Optimal coordinates: (-10, -10).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return np.sqrt(np.abs(np.cos(np.sqrt(np.abs(x[:, 0] ** 2 + x[:, 1] ** 2))))) + 0.01 * (x[:, 0] + x[:, 1])

def mishra5(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d = 2.
            Optimal function value: −1.019829.
            Optimal coordinates: (−1.98682, −10).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return (np.sin((np.cos(x[:, 0]) + np.cos(x[:, 1])) ** 2) ** 2
            + np.cos((np.sin(x[:, 0]) + np.sin(x[:, 1])) ** 2) ** 2 + x[:, 0]) ** 2 + 0.01 * (x[:, 0] + x[:, 1])

def mishra6(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-10, 10].
            Dimensions: d = 2.
            Optimal function value: −2.28395.
            Optimal coordinates: (2.88631, 1.82326).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    a = (np.cos(x[:, 0]) + np.cos(x[:, 1])) ** 2
    b = (np.sin(x[:, 0]) + np.sin(x[:, 1])) ** 2
    c = 0.1 * ((x[:, 0] - 1) ** 2 + (x[:, 1] - 1) ** 2)
    return c - np.log((np.sin(a) ** 2 - np.cos(b) ** 2 + x[:, 0]) ** 2)

def dropwave(x):
    """ Landscape propierties: Multimodal, Continuous.
            Domain: [-5.12, 5.12].
            Dimensions: d = 2.
            Optimal function value: -1.
            Optimal coordinates: (0, 0).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return - (1 + np.cos(12 * np.sqrt(np.sum(x**2, axis=1)))) / (0.5 * (np.sum(x**2, axis=1)) + 2)

def hosaki(x):
    """ Landscape propierties: Non-Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [0, 10].
            Dimensions: d = 2.
            Optimal function value: −2.345811.
            Optimal coordinates: (4, 2).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return (1 - 8 * x[:, 0] + 7 * x[:, 0] ** 2 - (7 / 3) * x[:, 0] ** 3 + - (1 / 4) * x[:, 0] ** 4) \
           * (x[:, 1] ** 2) * np.exp(-x[:, 0])
		   
def damavandi(x):
    """ Landscape propierties: Non_Separable, Non-Scalable, Differentiable, Multimodal, Continuous.
            Domain: [0, 14].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: (2,2).
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return (1 - np.abs((np.sin(np.pi * (x[:, 0] - 2)) * np.sin(np.pi * (x[:, 1] - 2))) /
                       (np.pi ** 2 * ((x[:, 0] - 2) * (x[:, 1] - 2)))) ** 5) \
           * (2 + (x[:, 0] - 7) ** 2 + 2 * (x[:, 1] - 7) ** 2)

def parsopoulos(x):
    """ Landscape propierties: Separable, Scalable, Differentiable, Multimodal, Continuous.
            Domain: [-5, 5].
            Dimensions: d = 2.
            Optimal function value: 0.
            Optimal coordinates: 12 different pair coordinates.
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return  np.cos(x[:, 0]) ** 2 + np.sin(x[:, 1]) ** 2

def vincent(x):
    """ Landscape propierties: Multimodal.
            Domain: [0.25, 10].
            Dimensions: d >= 2.
            Optimal function value: -d.
            Optimal coordinates: x_i = 7.70628098.
            Arguments:
            x -- A population of d-dimensional vector coordinates.
    """
    return -np.sum(np.sin(10 * np.log(x)), axis=1)

#################################
## Binary problems (v1.0, SRG) ##
#################################

def oneMax(B):
    """ oneMax counts the numbers of bits set in a given bitstring. Its maximum occurs when all d bits are set to 1.
        NB. Since optimiser minimises by default, the sum is negated for maximisation purposes.

        Arguments:
        B -- a population of bitstrings of length d
    """
    return -np.sum(B, axis=1)

def squareWave(B):
    """ squareWave computes the similarity of a bitstring to a periodic waveform of length d and period tau
        consisting of instantaneous transitions between levels 0 and 1 every tau/2 bits. It is basically a discrete
        version of a sin waveform, indeed it can be think of as the sign of the sin.
        NB. Since optimiser minimises by default, the similarity is negated for maximisation purposes.

        Arguments:
        B -- a population of bitstrings of length d. It is suggested d being a perfect square so that tau = sqrt(d)
    """
    d = B.shape[1]; tau = int(np.sqrt(d))
    S = -1 * (2 * (np.arange(d) // tau) - (2 * np.arange(d) // tau))
    return np.array([-np.sum(B_ == S) for B_ in B])

def binVal(B):
    """ binVal obtains the decimal value of a given bitstring. Its maximum occurs when all d bits are set to 1.
        NB. Since optimiser minimises by default, the value is negated for maximisation purposes.

        Arguments:
        B -- a population of bitstrings of length d
    """
    d = B.shape[1]
    # In the following line, use dtype=float to avoid overflow when d>=64 bits
    return -np.sum(np.multiply(2**np.arange(d, dtype=np.float)[::-1], B), dtype=np.float, axis=1)

def powSum(B):
    """ powSum obtains the sum of the exponents of the powers of two (or loci) that are set to one in a bitstring.
        Its maximum occurs when all d bits are set to 1.
        NB. Since optimiser minimises by default, the value is negated for maximisation purposes.

        Arguments:
        B -- a population of bitstrings of length d
    """
    d = B.shape[1]
    # In the following line, use dtype=float to avoid overflow when d>=64 bits
    return -np.sum(np.multiply(np.arange(1, d+1, dtype=np.float)[::-1], B), dtype=np.float, axis=1)


########################################
## Combinatorial problems (v2.0, SRG) ##
########################################

# Knapsack Problem #

def knapsack_instance(filename):
    """ knapsack_instance loads from a file the parameters of an instance of KP problem
        The data is stored in a global dictionary.
        Arguments:
        filname -- the name of the file with the instance data

        Output:
        kp_data -- a global dictionary containing the parameters of a KP instance
    """
    file_ = open(filename)
    line_ = file_.readline().split()
    n_items, C_max = int(line_[0]), float(line_[1])
    data_ = np.loadtxt(filename, skiprows=1, dtype=float)
    profits = np.array(data_[:, 0])
    weights = np.array(data_[:, 1])

    global kp_data
    kp_data = {"n_items": n_items, "C_max": C_max, "weights": weights, "profits": profits}
    print("KP instance loaded!", kp_data)
    return kp_data


def knapsack_discard(B):
    """ knapsack_discard obtains fitnesses using the KP cost function, except on those candidates violating
        the capacity constraint. The unfeasible candidates are "discarded" by setting their fitness to -Inf.
        NB. Since optimiser minimises by default, the value is negated for maximisation purposes.

        Arguments:
        B -- a population of bitstrings of length d
        kp_data -- a global dictionary containing the parameters of a KP instance
    """
    # First we need to retrieve the KP instance #
    global kp_data
    C_max = kp_data["C_max"]
    weights = kp_data["weights"]
    profits = kp_data["profits"]

    # Now compute the costs obtained with each candidate
    C = np.sum(np.multiply(weights, B), axis=1)  # Capacity of all candidates
    P = np.sum(np.multiply(profits, B), axis=1)  # Profitability of all candidates
    discard_mask = (C > C_max)                   # Filter out those violating the C_max constraint
    P[discard_mask] = -np.Inf                  # Discard them
    return -P                                    # Return profits

# End of file #