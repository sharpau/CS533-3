__author__ = 'Austin'

import operator
import math
import random


# bandit algorithms take in a bandit and a budget of arm pulls and return the best arm
def incremental_uniform(bandit, budget):
    pulls = [0 for _ in range(bandit.get_num_arms())]
    reward = [0 for _ in range(bandit.get_num_arms())]

    current_arm = 0
    for i in range(budget):
        reward[current_arm] += bandit.pull(current_arm)
        pulls[current_arm] += 1
        current_arm = (current_arm + 1) % bandit.get_num_arms()

    max_index, _ = max(enumerate([reward[x] / pulls[x] for x in range(bandit.get_num_arms())]), key=operator.itemgetter(1))

    return max_index


def ucb(bandit, budget):
    pulls = [0 for _ in range(bandit.get_num_arms())]
    reward = [0 for _ in range(bandit.get_num_arms())]

    for i in range(budget):
        # use heuristic to balance exploration and pulling best arm, for min. cumulative regret
        heuristic = []
        for j in range(bandit.get_num_arms()):
            if pulls[j] == 0:
                heuristic.append(float("inf"))
            else:
                heuristic.append(reward[j] / pulls[j] + math.sqrt(2 * math.log(i) / pulls[j]))

        # pull arm with max heuristic
        current_arm, _ = max(enumerate(heuristic), key=operator.itemgetter(1))
        reward[current_arm] += bandit.pull(current_arm)
        pulls[current_arm] += 1

    max_index, _ = max(enumerate([reward[x] / pulls[x] for x in range(bandit.get_num_arms())]), key=operator.itemgetter(1))

    return max_index


def epsilon_greedy(bandit, budget, epsilon):
    assert(epsilon > 0)
    assert(epsilon < 1)

    pulls = [0 for _ in range(bandit.get_num_arms())]
    reward = [0 for _ in range(bandit.get_num_arms())]

    for i in range(budget):
        avg = []
        for j in range(bandit.get_num_arms()):
            if pulls[j] == 0:
                avg.append(0.0)
            else:
                avg.append(reward[j] / pulls[j])
        best, _ = max(enumerate(avg), key=operator.itemgetter(1))
        arm = -1

        if random.random() > epsilon:
            # select random non-best arm
            nonbest = range(bandit.get_num_arms())
            nonbest.remove(best)
            arm = random.choice(nonbest)
        else:
            # select best arm
            arm = best

        reward[arm] += bandit.pull(arm)
        pulls[arm] += 1
        
    max_index, _ = max(enumerate([reward[x] / pulls[x] for x in range(bandit.get_num_arms())]), key=operator.itemgetter(1))

    return max_index
