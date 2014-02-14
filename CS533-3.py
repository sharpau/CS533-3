__author__ = 'Austin'

import bandit
import algorithms


def cumulative_regret(bandit, pulls):
    max_expected = max([bandit.rewards[i] * bandit.probabilities[i] for i in range(bandit.get_num_arms())])
    regrets = [0 for _ in range(len(pulls))]
    for i in range(len(pulls)):
        if i > 0:
            idx = pulls[i]
            regrets[i] = regrets[i - 1] + (max_expected - (bandit.rewards[idx] * bandit.probabilities[idx]))
        else:
            idx = pulls[i]
            regrets[i] = max_expected - (bandit.rewards[idx] * bandit.probabilities[idx])

    return regrets


p1 = [1.0 for x in range(9)] + [0.1]
r1 = [0.05 for x in range(9)] + [1.0]
b1 = bandit.Bandit(p1, r1)

p2 = [0.1 for x in range(20)]
r2 = [float(x) / 20.0 for x in range(1, 21)]
b2 = bandit.Bandit(p2, r2)

# a bandit where all expected rewards are equal, but become more infrequent for higher reward arms
p3 = [0.1 / float(x) for x in range(1, 21)]
r3 = [float(x) for x in range(20, 0, -1)]
b3 = bandit.Bandit(p3, r3)

best, dist, history = algorithms.incremental_uniform(b1, 10000)
reg = cumulative_regret(b1, history)
best, dist, history = algorithms.ucb(b1, 10000)
reg = cumulative_regret(b1, history)
best, dist, history = algorithms.epsilon_greedy(b1, 10000, 0.5)
reg = cumulative_regret(b1, history)