'''
A test script showcasing different examples of solving 
optimisation problems with the FLW algorithm.

Copyright (c) 2023 Sergio Rojas-Galeano

Version: 1.0 (April/2023)
License: GPLv3 (see https://www.gnu.org/licenses/gpl-3.0.txt)
'''

## Import libraries ##
from flw_class import FLW_Real
from flw_bm import * 
import time
import numpy as np

def flw_test(test_case: str) -> None:
    """
    A function to test the FLW algorithm on a variety of cases.

    Parameters
    ----------
    test_case : str
        Test case identifier. Can be one of the following:
        - "2D": run experiments on a set of 2D problems.
        - "HD": run experiments on higher-dimensional problems.
        - Any cost function name: optimise the function and visualizes the results.

    Returns
    -------
    None
    """
    # Set random generator seed 
    np.random.seed(int(str(int(time.time() * 1000))[-8:-1]))  

    match test_case:
        # 2D experiments
        case "2D":
            nreps = 10              # Number of repetitions per problem            
            problems = [sphere, 
                        rastrigin, 
                        rastrigin_offset, 
                        rastrigin_bipolar, 
                        rosenbrock, 
                        himmelblau,  # himmelblau is only defined for 2D
                        ]
    
            # Optimise each problem, print results for a number of reps
            for problem in problems:
                for _ in range(nreps):
                    flw = FLW_Real(fcost=problem) 
                    flw.optimise()
                    flw.summary()
    
            # eggholder is only defined for 2D, with different bounds
            for _ in range(nreps):
                flw = FLW_Real(fcost=eggholder, 
                               LB=np.array([-512., -512.]), 
                               UB=np.array([512., 512.]), 
                               resets=50)
                flw.optimise()
                flw.summary()
    
        # Higher-dimension experiments
        case "HD":
             # Adjust these settings to your preference
            nreps = 5              # Number of repetitions per problem
            d = 10                 # Dimensions for search space (must be even!)
            max_evals = 10000 * d  # Maximum number of evaluations
            resets = 40            # Number of resets per algorithm run
            setup = "uniform"      # Setup strategy ('latin_hypercube'|'uniform')
            sampler = "uniform"    # Walk-step sampling strategy ('uniform'|'adaptive')
            elite = "any"          # Elitism strategy ('any'|'walkers'|'everyone')
            operators = "FWRE"     # Search operators ('FWRE'|'FER'|'WER'|'FE'|etc)
    
            problems = [sphere, 
                        rastrigin, 
                        rastrigin_offset, 
                        rastrigin_bipolar, 
                        rosenbrock, 
                        ]
            UB = np.repeat(5., d); LB = -UB
            
            for problem in problems:
                for _ in range(nreps):
                    flw = FLW_Real(fcost=problem, d=d,
                                   UB=UB, LB=LB,
                                   max_evals=max_evals, 
                                   resets=resets, 
                                   setup=setup,
                                   sampler=sampler, 
                                   elite=elite, 
                                   operators=operators
                                    ) 
                    flw.optimise()
                    flw.summary()
                    
        # Otherwise use parameter as problem and perform visualization
        case _:
            flw = FLW_Real(fcost=eval(test_case), 
                           max_evals=2000, 
                           viz=True, hist=True) 
            flw.optimise()
            flw.summary()
            flw.plotBest()
    

if __name__ == "__main__":
    flw_test("sphere")

# End of file #