__author__ = 'Austin'

import operator
import math
import random


# bandit algorithms take in a bandit and a budget of arm pulls
# and return distribution of pulls, history of pulls (ordered), history of best-looking arm

def get_expected(reward, pulls):
    if pulls == 0:
        return 0
    else:
        return reward / pulls

# pulls each arm in order
def incremental_uniform(bandit, budget):
    pulls = [0 for _ in range(bandit.get_num_arms())]
    reward = [0 for _ in range(bandit.get_num_arms())]
    history = [0 for _ in range(budget)]
    best = [0 for _ in range(budget)]

    current_arm = 0
    for i in range(budget):
        reward[current_arm] += bandit.pull(current_arm)
        pulls[current_arm] += 1
        current_arm = (current_arm + 1) % bandit.get_num_arms()
        history[i] = current_arm

        best[i], _ = max(enumerate([get_expected(reward[x], pulls[x]) for x in range(bandit.get_num_arms())]), key=operator.itemgetter(1))

    return pulls, history, best


# tries to minimize cumulative regret, balances exploration and exploitation
def ucb(bandit, budget):
    pulls = [0 for _ in range(bandit.get_num_arms())]
    reward = [0 for _ in range(bandit.get_num_arms())]
    history = [0 for _ in range(budget)]
    best = [0 for _ in range(budget)]

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
        history[i] = current_arm
        best[i], _ = max(enumerate([get_expected(reward[x], pulls[x]) for x in range(bandit.get_num_arms())]), key=operator.itemgetter(1))

    return pulls, history, best


# epsilon chance of pulling best arm so far, otherwise pull random arm
def epsilon_greedy(bandit, budget, epsilon):
    assert(epsilon > 0)
    assert(epsilon < 1)

    pulls = [0 for _ in range(bandit.get_num_arms())]
    reward = [0 for _ in range(bandit.get_num_arms())]
    history = [0 for _ in range(budget)]
    best = [0 for _ in range(budget)]

    for i in range(budget):
        best_arm, _ = max(enumerate([get_expected(reward[x], pulls[x]) for x in range(bandit.get_num_arms())]), key=operator.itemgetter(1))
        arm = -1

        if random.random() > epsilon:
            # select random non-best arm
            nonbest = range(bandit.get_num_arms())
            nonbest.remove(best_arm)
            arm = random.choice(nonbest)
        else:
            # select best arm
            arm = best_arm

        reward[arm] += bandit.pull(arm)
        pulls[arm] += 1
        history[i] = arm
        best[i] = best_arm

    return pulls, history, best
