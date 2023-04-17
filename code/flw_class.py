"""
FLW (Follow-the-Leader + Walkers) is a multi-agent search algorithm for
optimising bound-constrained cost functions. Its search strategy is based on
follow-the-leader intensification and random walk diversification rules. 
This module consist of FLW_Real class, an optimiser for real-valued functions.

Copyright (c) 2023 Sergio Rojas-Galeano

Version: 1.0 (April/2023)
License: GPLv3 (see https://www.gnu.org/licenses/gpl-3.0.txt)
"""

## Python 2.7 compatibility ##
from __future__ import print_function
import matplotlib
matplotlib.use('TkAgg')

## Required libraries ##
import numpy as np
import time
import matplotlib.pyplot as plt
from flw_bm import *
from scipy.stats import qmc


class FLW_Real():
    """ 
    Class implementation of the FLW algorithm for real-valued (continuous) optimisation 
    """
 
    def __init__(self, fcost=sphere, d=2, LB=np.array([-5.,-5.]), 
                 UB=np.array([5.,5.]), n=20, nw=.20,
                 max_evals=10000, resets=40, setup='uniform',
                 sampler='uniform', elite='any', operators='FWER',
                 viz=False, hist=False):
        """
        Initialise a Follow-the-Leader + Walkers (FLW) optimiser.
    
        Parameters
        ----------
        fcost : callable, optional
            Cost function to be optimized. The function should take a numpy array of 
            shape (n,d) as input, where n is the number of agents and d is the 
            dimension of the search space. The function should return an array of 
            shape (n,) with the corresponding cost values. See the companion file 
            'flw_bm.py' for a benchmark of function costs. Default is the 'sphere'
            cost funnction.
        d : int, optional
            Dimension of the search space. Default is 2 dimensions.
        LB : numpy.ndarray, optional
            Array of shape (d,) with the lower bounds of the search space.
        UB : numpy.ndarray, optional
            Array of shape (d,) with the upper bounds of the search space.
        n : int, optional
            Number of agents in the population.
        nw : float, optional
            Percentage of agents to be walkers.
        max_evals : int, optional
            Maximum number of function evaluations allowed.
        resets : int, optional
            Number of times the population is reset to new initial positions.
        setup : str, optional
            Method used to sample the initial positions of the agents within
            the search space. 
            Possible values are 'latin_hypercube' and 'uniform'.
        sampler : str, optional
            Method used to sample the width parameter (sigma) of walkers' steps. 
            Possible values are 'adaptive' and 'uniform'.   
        elite : str, optional
            Determines which group of agents is passed on as elite to the next 
            iteration, maintaining the global best solution discovered so far. 
            Possible values are:
            - 'any': Any individual (a walker) is selected as elite.
            - 'walkers': All the walkers are selected as elite.
            - 'everyone': All the agents are selected as elite.   
        operators : str, optional
            A string of character flags representing which search operators will 
            be used to optimise the cost function. Any combination of the characters
            'F', 'W', 'E', and 'R' is valid, each character representing the 
            following operator:
            - 'F': Follow-the-leader intensification operator
            - 'W': Random walk diversification operator
            - 'E': Elitism operator, according to the defined 'elite' strategy
            - 'R': Reset operator, according to the 'resets' parameter.
            For example, operators='FWER' indicates that the algorithm will apply 
            all the search operators, while operators='REW' will apply only reset, 
            elitism, and walk operators.   
        viz : bool, optional
            If True, visualize the optimization process using a scatter plot.
            Only valid for 2D problems (d=2). Default is False.
        hist : bool, optional
            If True, store history of best, average and worst solutions found during
            the optimization process. Default is False.
    
        Returns
        -------
        None
        """            
        self.fcost = fcost           
        self.d = d                   
        self.LB = LB                 
        self.UB = UB                 
        self.n = n                   
        self.nw = nw                 
        self.max_evals = max_evals   
        self.evals = 0                   # Counter of evaluations of cost function
        self.resets = resets
        self.setup = setup
        self.sampler = sampler
        self.sigmas = 10**np.arange(-4, 3, dtype=float)  # Array of step widths for walkers move
        self.counts = np.ones(len(self.sigmas))          # Count histogram of successful walkers move       
        self.operators = operators
        self.elite = elite
        self.xbest = np.empty(self.d)    # The best solution found
        self.fbest = np.Inf              # The cost of best solution found
        self.ibest = np.Inf              # The iteration were best was found
        self.toc = 0                     # Timing counter
        self.viz = viz               
        self.hist = hist           
        self.fmins = []; self.favgs = []; self.fmaxs = [] # History of solutions values

    def create(self):
        """
        Create a population of agents and assign roles for the FLW algorithm.

        Returns:
        -------        
        Tuple containing:
            - P (ndarray): A numpy array of shape (n, d) representing the coordinates
              of the population of agents within the search space.
            - followers (list): A list of indices indicating the followers in the population.
            - walkers (list): A list of indices indicating the walkers in the population.
            - leader (int): An integer indicating the index of the leader in the population.
        
        Notes:
        ------        
        This function initialises the population of agents used by the FLW algorithm and
        assigns randomly the roles of follower, walker, and leader to the agents. The 
        population is generated using one of two methods depending on the 'setup' parameter: 
        uniform sampling or Latin Hypercube sampling. The P ndarray variable will hold
        the initial positions of the agents.
        """
        P = np.zeros((self.n, self.d))
        if self.setup == "uniform":
            for j in range(self.d):
                P[:, j] = np.random.uniform(self.LB[j], self.UB[j], self.n).reshape((1, self.n))
        if self.setup == "latin_hypercube":
            lhc = qmc.LatinHypercube(self.d)
            sample = lhc.random(n=self.n)
            P = qmc.scale(sample, self.LB, self.UB)
        followers, walkers, leader = self.setRoles()
        return P, followers, walkers, leader

    def setRoles(self):
        """
        Randomly assign roles to the agents in the population.
        
        Returns
        -------
        Tuple containing:
            - followers: an array with the indices of the followers in the population
            - walkers: an array with the indices of the walkers in the population
            - leader: an integer representing the index of the leader in the population
        """
        walkers = np.random.choice(self.n, int(self.n * self.nw), replace=False)
        followers = np.setdiff1d(range(self.n), walkers)
        leader = np.random.choice(followers)
        return followers, walkers, leader
        
    def follow(self, X, Xl):
        """
        Apply the Follow-the-Leader intensification operator to the solution set X.

        Parameters
        ----------
        X : numpy.ndarray
            The solution set to be modified.
        Xl : numpy.ndarray
            The array of coordinates of the leader at the current iteration.

        Returns
        -------
        X : numpy.ndarray
            The modified solution set.
        F : numpy.ndarray
            The values of the cost function for the modified solution set.

        Notes
        -----
        This function updates the position of each follower by moving it closer to 
        the leader. The movement update is proportional to a random factor times
        the distance of the agent and the leader along each coordinate axis.
        """
        # Iterate over each follower (len(X) is the number of followers)
        for i in range(len(X)):  
            # iterate over each coordinate
            for j in range(self.d):  
                deltax = Xl[j] - X[i, j]
                X[i, j] = np.clip(X[i, j] + (3 * np.random.rand() * (deltax)), self.LB[j], self.UB[j])
        
        
        # Update the cost function value for all followers
        F = self.evaluate(X)         
        return X, F

    def walk(self, X):
        """
        Apply the random walk diversification operator to the solution set X.
    
        Parameters
        ----------
        X : numpy.ndarray
            The solution set to be modified.
    
        Returns
        -------
        X : numpy.ndarray
            The modified solution set.
        F : numpy.ndarray
            The cost values of the modified solution set.
    
        Notes
        -----
        The walk operator updates each solution by perturbing a random 
        coordinate with a Gaussian-distributed step size. The step size 
        depends on a random standard deviation chosen from the 'sigmas' 
        array. The selection is based on the 'sampler' strategy, which 
        can be set to 'uniform' or 'adaptive'. In the latter case, the 
        sigma value with the highest successful counter has a higher 
        probability of being chosen.
        
        The number of successful sigma-step counters is updated every 
        time a walker improves over the current best solution with a 
        particular sigma used to sample the random walk step. In this
        way, an adaptive step size operator for the particular cost
        function can be implemented.
        """
        F = np.zeros(len(X))

        # Iterate over each walker (len(X) is the number of walkers)
        for i in range(len(X)):
            # Choose one random coordinate to update
            j = np.random.choice(self.d)  
            if self.sampler == "uniform":
                winner = np.random.choice(len(self.sigmas))
            elif self.sampler == "adaptive":
                winner = self.roulette(self.counts)
            X[i, j] = np.clip((X[i, j] + self.sigmas[winner] * np.random.randn()), self.LB[j], self.UB[j])
            # X[i, j] = np.clip((X[i, j] + self.sigmas[winner] * self.randLevy(1.9)), self.LB[j], self.UB[j])
            
            # Update the cost function value for each walker
            F[i] = self.evaluate(np.expand_dims(X[i], 0))

            # Update counters of succesul walk moves
            self.counts[winner] += (1 if F[i]<self.fbest else 0)                
        return X, F

    def reset(self):
        """
        Reset the FLW algorithm by creating a new population and resetting the 
        successful sigma-step counters.

        Returns
        -------
        A tuple containing:
            - P (ndarray): A numpy array of shape (n, d) representing the coordinates
              of the population of agents within the search space.
            - followers (list): A list of indices indicating the followers in the population.
            - walkers (list): A list of indices indicating the walkers in the population.
            - leader (int): An integer indicating the index of the leader in the population.
        """
        P, followers, walkers, leader = self.create() 
        self.counts = np.ones(len(self.sigmas)) 
        return P, followers, walkers, leader

    def getLeader(self, F):
        """
        Return the index of the leader, i.e. agent with the lowest cost value 
        in the current population.
        NB: The algorithm minimises by default (argmin)
    
        Parameters
        ----------
        F : numpy.ndarray
            The function cost values for the solution set.
            
        Returns
        -------
        int
            The index of the leader agent.
        """    
        return np.argmin(F, axis=0)

    def gpm(self, P):
        """
        Genotype-to-phenotype mapping function.
        
        This function allows the user to implement a genotype-to-phenotype mapping
        of the set of solutions P. By default, it performs a direct mapping,
        returning the genotype array P without any transformations or modifications.
    
        Parameters
        ----------
        P : ndarray
            A numpy array of shape (n, d) representing the genotype, 
            in this case the coordinates of the population of agents within 
            the search space.
    
        Returns
        -------
        P : ndarray 
            The phenotype array of shape (n, d) resulting from the mapping 
            of the input genotype array.
    
        """    
        return P                                   

    def evaluate(self, X):
        """
        Evaluate the values of a set of solutions X using the defined cost function 'fcost'.
        This function also updates the number of evaluations performed on the cost function.

        Parameters
        ----------
        X : numpy.ndarray
            The solution set to be evaluated.
    
        Returns
        -------
        costs : numpy.ndarray
            The values of the cost function for the input solution set.    
        """
        costs = self.fcost(self.gpm(X))
        self.evals += len(costs)
        return costs
        
    def optimise(self):
        """
        Optimises the cost function using the FLW algorithm.
        
        Notes
        -----
        - The function performs a loop that continues until the maximum number 
          of evaluations is reached. The evaluations counter is updated every
          time the cost function is evaluated.
        - At each iteration, the function moves the agents in the population 
          according to the selected search operators:
            - 'F' follow
            - 'W' walk
            - 'E' elitism
            - 'R' reset
          These blocks of code are executed or not, at each iteration.
        - The loop also updates the best solution found so far, if a leader 
          improves the value of the cost function.
        - If enabled, the function tracks the history of the best, average, and 
          worst solutions, every certain number of evaluations.
        - If enabled, the function visualizes the optimization process using a 
          scatter plot, every certain number of evaluations (for 2D problems only).
        
        Returns
        -------
        None
        """
        # Initialise visualisation settings
        if self.viz: 
            self.vizSetup()
        
        # Start execution timer
        tic = time.time()
        
        # Create initial population and set any initial solution
        P, followers, walkers, leader = self.create()
        F = self.evaluate(P)
        self.fbest, self.xbest = F[leader], P[leader]  
        
        # This is the optimisation loop.
        # Note that the evals counter is incremented with every call to the evaluate() 
        # function, including calls made within the follow() and walk() functions.
        while self.evals < self.max_evals:   
            
            # Follow-the-leader operator
            if 'F' in self.operators:
                P[followers], F[followers] = self.follow(P[followers], P[leader])  
            
            # Random walk operator
            if 'W' in self.operators:
                P[walkers], F[walkers] = self.walk(P[walkers])  
                
            # Update current leader
            leader = self.getLeader(F)  
            
            # Update best solution found so far ('<' for minimisation)
            if F[leader] < self.fbest:  
                self.fbest, self.xbest, self.ibest = F[leader], np.copy(P[leader]), self.evals         
            
            # Elitism operator
            if 'E' in self.operators:
                # Any walker is elite
                if self.elite == "any":         
                    P[walkers[0]], F[walkers[0]] = np.copy(self.xbest), self.fbest 
                
                # All walkers are elite
                elif self.elite == "walkers":  
                    for walker in walkers:       
                        P[walker], F[walker] = np.copy(self.xbest), self.fbest    
                
                # All agents are elite
                elif self.elite == "everyone":
                    for i in range(self.n):     
                        P[i], F[i] = np.copy(self.xbest), self.fbest              
            
            # Reset operator
            if 'R' in self.operators and not self.evals % (self.max_evals/self.resets):   
                P, followers, walkers, leader = self.reset()  
                F = self.evaluate(P)
            
            # Track history of solutions
            if self.hist and not (self.evals % (self.max_evals/500)):  
                self.fmins.append(self.fbest)
                self.favgs.append(np.mean(F));
                self.fmaxs.append(F[np.argmax(F, axis=0)])
            
            # If visualisation, do it every 200 evaluation
            if self.viz and not (self.evals % 200):   
                self.vizIteration(self.evals, P, followers, walkers, leader)
            
        # Stop timer and record elapsed time 
        self.toc = time.time() - tic

    def roulette(self, weights):
        """
        Randomly select a winner from an array of weights using the roulette 
        wheel selection method, where the weights are treated as probabilities.
    
        Parameters
        ----------
        weights : array-like
            A 1D array-like with the weights of each element in the array.
    
        Returns
        -------
        int
            The index of the selected winner.
        """
        winner = np.random.choice(len(weights), p=(weights / np.sum(weights))) 
        return winner

    def randLevy(self, beta):
        """
        Generate a random sample from a Levy distribution with the given
        shape parameter beta.
    
        Parameters
        ----------
        beta : float
            The shape parameter of the Levy distribution. NB: 1 < beta < 2.
    
        Returns
        -------
        float
            A random sample from the Levy distribution.
        """
        num = np.random.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        den = np.random.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        std = (num / den) ** (1 / beta)
        u = np.random.normal(0, std * std)
        v = np.random.normal()
        return u / (abs(v) ** (1 / beta))

    def summary(self):
        """
        Print a summary of the optimisation results, including the problem name,
        dimensionality, elapsed time, best cost found, number of evaluations needed, 
        and the coordinates of the best solution discovered.
    
        Returns
        -------
            None
        """
        print("\n%s RESULTS %s\nProblem: %s (d=%d)\nEllapsed time: %.2fs \
               \nBest cost: %.20f \nFound after: %d evaluations \n%s"    \
              % ("-"*30, "-"*30, self.fcost.__name__, self.d, self.toc,  \
                 self.fbest, self.ibest, "-"*70))
        print("Best solution (genotype): ", list(map(float, ["%.3f" % v for v in self.xbest])))
        print("Best solution (phenotype): ", self.gpm(self.xbest), "\n", "-"*70, "\n\n", sep="")

    def getResults(self):
        """
        Return the optimisation results obtained by the FLW algorithm.

        Returns
        -------
        A tuple containing the following optimisation results:
            - str: Name of the cost function being optimised.
            - int: Dimensionality of the optimisation problem.
            - float: Time elapsed since the start of the execution (seconds).
            - float: Cost function value of the best solution found.
            - int: Iteration number where the best solution was found.
            - array-like: The coordinates of the best solution found.
        """
        return self.fcost.__name__, self.d,self.toc, self.fbest, self.ibest, self.xbest

    def getHist(self):
        """
        Get the history of best, average and worst function values found during 
        the optimisation process.
        
        Returns
        -------
        A tuple containing the history of the following values:
            - array-like: cost function value of best solution per iteration.
            - array-like: average cost function value of solutions per iteration.
            - array-like: cost function value of worst solution per iteration.
        """
        return self.fmins, self.favgs, self.fmaxs

    def plotBest(self):
        """
        Show a plot with the history of the best solutions discovered
        throughout the optimisation process, if history was enabled.
        
        Returns
        -------
        None
        """
        if self.hist:
            plt.plot(self.fmins)
            plt.ylim(min(0, self.fbest), np.mean(self.fmins))
            plt.title("fbest plot (problem: %s)" % self.fcost.__name__)
            plt.show()
        else:
            print("Sorry, history was not enabled, data unavailable.")
    
    def vizSetup(self):
        """
        Set up the X, Y an Z meshgrid values to visualise a 2D surface of the cost function.
        
        Returns
        -------
        None
        """
        X = np.linspace(self.LB[0], self.UB[0], 100)
        Y = np.linspace(self.LB[1], self.UB[1], 100)
        self.X, self.Y = np.meshgrid(X, Y)
        self.Z = self.fcost(np.vstack([self.X.flatten(), self.Y.flatten()]).T).reshape(self.X.shape)

    def vizIteration(self, i, P, followers, walkers, leader):
        """
        For the current iteration, plot the 2D surface of the cost function, 
        with the population of agents (solutions) scattered around.
        
        Returns
        -------
        None
        """ 
        plt.contourf(self.X, self.Y, self.Z, 8, colors=('navy', 'royalblue', 'skyblue', 'greenyellow', 'yellow', 'darkorange', 'tomato', 'crimson', 'maroon'))
        plt.title("Problem: %s / Evaluations: %d / Best cost so far: %.6f  " % (self.fcost.__name__, i, self.fbest))
        plt.scatter(P[followers, 0], P[followers, 1], marker='^', c='black')
        plt.scatter(P[walkers, 0], P[walkers, 1], marker='d', c='darkgreen')
        plt.scatter(P[leader, 0], P[leader, 1], marker='o', c='red')
        plt.scatter(self.xbest[0], self.xbest[1], marker='*', c='white')
        plt.xlim(self.LB[0], self.UB[0]); plt.ylim(self.LB[1], self.UB[1])
        plt.draw(); plt.pause(10); plt.clf()

## End of class ##