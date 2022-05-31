import numpy as np
from scipy import rand
from math import exp
from math import exp
from matplotlib import pyplot
from numpy.random import seed
# source https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/

# objective function
def objective(x):
    return x[0]**2.0


# simulated annealing algorithm
def simulated_annealing(objective, bounds, n_iteration, step_size, temp):
    # generate an initial point
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])

    # evaluate the initial point
    best_eval = objective(best)

    # current working solution
    curr, curr_eval = best, best_eval
    scores = list()

    # run the algorithm
    for i in range(n_iteration):
        # take a step
        candidate = curr + rand(len(bounds)) * step_size
        print(rand(len(bounds)))
        # evaluate candidate point
        candidate_eval = objective(candidate)

        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # keep track of scores
            scores.append(best_eval)
            # report process
            # print('>%d f(%s) =  %.5f' % (i, best, best_eval))

        # difference between candidate and current point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)

        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return best, best_eval, scores


# seed the pseudorandom number generator
seed(1)
# define range for input
bounds = np.asarray([[-5.0, 5.0]])
s = rand(len(bounds))
print(s)
# define the total iterations
n_iterations = 1000
# define the maximum step size
step_size = 0.1
# initial temperature
initial_temp = 10


# perform the simulated annealing search
best, score, scores = simulated_annealing(objective, bounds, n_iterations, step_size, initial_temp)
print('Done!')
print('f(%s) = %f' % (best, score))

# line plot of best scores
pyplot.plot(scores, '.-')
pyplot.xlabel('Improvement Number')
pyplot.ylabel('Evaluation f(x)')
pyplot.show()


