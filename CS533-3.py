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

def simple_regret(bandit, best):
    max_expected = max([bandit.rewards[i] * bandit.probabilities[i] for i in range(bandit.get_num_arms())])
    regrets = [0 for _ in range(len(best))]
    for i in range(len(best)):
        idx = best[i]
        regrets[i] = max_expected - (bandit.rewards[idx] * bandit.probabilities[idx])

    return regrets



p1 = [1.0 for x in range(9)] + [0.1]
r1 = [0.05 for x in range(9)] + [1.0]
b1 = bandit.Bandit(p1, r1)

p2 = [0.1 for x in range(20)]
r2 = [float(x) / 20.0 for x in range(1, 21)]
b2 = bandit.Bandit(p2, r2)

# 8 rewards are low, two are equal expected but unlikely
p3 = [0.1 for x in range(8)] + [0.5, 0.05]
r3 = [0.1 for x in range(8)] + [0.1, 1.0]
b3 = bandit.Bandit(p3, r3)

bandit_list = [b1, b2, b3]
b_num = 1

for b in bandit_list:
    trials = 100
    budget = 1000
    uniform_avg_cumulative = [0 for _ in range(budget)]
    uniform_avg_simple = [0 for _ in range(budget)]
    ucb_avg_cumulative = [0 for _ in range(budget)]
    ucb_avg_simple = [0 for _ in range(budget)]
    epsilon_avg_cumulative = [0 for _ in range(budget)]
    epsilon_avg_simple = [0 for _ in range(budget)]

    for _ in range(trials):
        # run a trial for each algorithm,
        # record simple and cumulative regret,
        dist, history, best = algorithms.incremental_uniform(b, budget)
        cumulative_reg = cumulative_regret(b, history)
        simple_reg = simple_regret(b, best)
        uniform_avg_cumulative = [x + y for x, y in zip(uniform_avg_cumulative, cumulative_reg)]
        uniform_avg_simple = [x + y for x, y in zip(uniform_avg_simple, simple_reg)]

        dist, history, best = algorithms.ucb(b, budget)
        cumulative_reg = cumulative_regret(b, history)
        simple_reg = simple_regret(b, best)
        ucb_avg_cumulative = [x + y for x, y in zip(ucb_avg_cumulative, cumulative_reg)]
        ucb_avg_simple = [x + y for x, y in zip(ucb_avg_simple, simple_reg)]

        dist, history, best = algorithms.epsilon_greedy(b, budget, 0.5)
        cumulative_reg = cumulative_regret(b, history)
        simple_reg = simple_regret(b, best)
        epsilon_avg_cumulative = [x + y for x, y in zip(epsilon_avg_cumulative, cumulative_reg)]
        epsilon_avg_simple = [x + y for x, y in zip(epsilon_avg_simple, simple_reg)]

    # have totals, average them
    results = [uniform_avg_cumulative, uniform_avg_simple, ucb_avg_cumulative, ucb_avg_simple, epsilon_avg_cumulative, epsilon_avg_simple]
    for idx, l in enumerate(results):
        results[idx] = [x / trials for x in l]

    # write results to csv
    with open("bandit" + str(b_num) + ".csv", "w") as out_file:
        out_file.write("uniform CR, uniform SR, ucb CR, ucb SR, epsilon CR, epsilon SR\n")
        for i in range(budget):
            for j in range(len(results)):
                out_file.write(str(results[j][i]) + ",")
            out_file.write("\n")
    b_num += 1



