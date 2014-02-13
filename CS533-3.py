__author__ = 'Austin'

import bandit
import algorithms

p1 = [1.0 for x in range(9)] + [0.1]
r1 = [0.05 for x in range(9)] + [1.0]
b1 = bandit.Bandit(p1, r1)

p2 = [0.1 for x in range(20)]
r2 = [float(x) / 20.0 for x in range(1, 21)]
b2 = bandit.Bandit(p2, r2)

p3 = [0.1 / float(x) for x in range(1, 21)]
r3 = [float(x) for x in range(1, 21)]
b3 = bandit.Bandit(p3, r3)

best, history = algorithms.incremental_uniform(b1, 10000)
best, history = algorithms.ucb(b1, 10000)
best, history = algorithms.epsilon_greedy(b1, 10000, 0.5)