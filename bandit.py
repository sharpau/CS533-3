__author__ = 'Austin'

import random


class Bandit(object):
    def __init__(self, probabilities, rewards):
        assert(len(probabilities) == len(rewards))
        self.probabilities = probabilities
        self.rewards = rewards
        random.seed()

    def get_num_arms(self):
        return len(self.probabilities)

    def pull(self, arm):
        if random.random() > self.probabilities[arm]:
            return 0
        else:
            return self.rewards[arm]

