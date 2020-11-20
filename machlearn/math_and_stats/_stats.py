# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt


class distance(object):

    def __init__(self, p=1):
        self.p = p # for Minkowski distance

    def Euclidean(self, x, y):
        """
        x is an array of (x_1, x_2, ..., x_n)
        y is an array of (y_1, y_2, ..., y_n)
        """
        distance = 0
        for d in range(len(x)):
            distance += (x[d] - y[d])**2
        distance **= 0.5
        return distance

    def Minkowski(self, x, y):
        """
        x is an array of (x_1, x_2, ..., x_n)
        y is an array of (y_1, y_2, ..., y_n)
        when p=2, it's the Euclidean distance
        """
        distance = 0
        for d in range(len(x)):
            distance += abs(x[d] - y[d])**self.p
        distance **= (1/self.p)
        return distance


class probability(object):
    """
    probabilities and distribution
    """

    def binomial_pmf(self, n_heads_in_results, n_coin_tosses, p_head=0.5):
        """
        n: number of independent experiments, each asking a yes-no question
        p: probability associated with outcome=yes
        """
        return stats.binom.pmf(k=n_heads_in_results, n=n_coin_tosses, p=p_head)

    def binomial_cdf(self, n_heads_in_results, n_coin_tosses, p_head=0.5):
        """
        n: number of independent experiments, each asking a yes-no question
        p: probability associated with outcome=yes
        """
        return stats.binom.cdf(k=n_heads_in_results, n=n_coin_tosses, p=p_head)

    def plot_binomial(self):
        n = 10
        k = np.arange(n+1)
        for y in [self.binomial_pmf(n_heads_in_results=k, n_coin_tosses=n),
                  self.binomial_cdf(n_heads_in_results=k, n_coin_tosses=n)]:
            plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            plt.bar(k, y)
            plt.xlabel('k')
            plt.plot(k, y, 'o-r')
            plt.show()

    def coin_toss(self, times=1, p_head=0.5):
        # 1 = Head, 0 = Tail
        return np.random.choice([0, 1], size=(times,), p=[p_head, 1-p_head])

    def fair_coin_toss(self, size=1):
        random_number_generated = np.array([])
        n_generated = 0
        while n_generated < size:
            p_head = random.uniform(0, 1)
            toss1 = self.coin_toss(times=1, p_head=p_head)
            toss2 = self.coin_toss(times=1, p_head=p_head)
            # the chance of (toss1='H' and toss2='T') is always the same as (toss1='T' and toss2='H'), regardless of what p_head is, since (p*(1-p)) = ((1-p)*p)
            # this is an example that with uncertainties, there are always some certainties.
            if toss1 == 1 and toss2 == 0:
                random_number_generated = np.append(random_number_generated, 1)
                n_generated += 1
            elif toss1 == 0 and toss2 == 1:
                random_number_generated = np.append(random_number_generated, 0)
                n_generated += 1
        return random_number_generated

    def head_tail_problem(self):
        """
        Problem:
            In a sequence of head and tail (e.g., HTHHHTTH...), the occurrences of HT and those of TH are always the same (or differ by at most just 1).
            This is true however you generate the head and tail sequence.

        Explanation:
            Imagine tail is like 1-min darkness in a room at the night. When the whole sequence is just tails, the room is completely dark during the whole right.
            Then at certain times during the night, the light was turned on (some tails are replaced by heads).
            Every time when the light was turned on, it involved [Tail -> Head] boarder condition.
            Unless the light was on the whole night, whenever it was turned off, it involved [Head -> Tail] boarder condition.
            Thus, whenever a head was inserted into the sequence, the leftmost boarder always involves a [Tail -> Head] and the rightmost boarder always involves a [Head -> Tail], and these come in pairs.
            That is why n_HT = n_TH±1
        """
        n_times = int(1e5)
        for generator in ['#1', '#2']:
            print(f"\nhead/tail sequence generator {generator}:")
            if generator == '#1':
                toss_outcomes = self.coin_toss(times=n_times, p_head=random.uniform(0, 1)).tolist()
            elif generator == '#2':
                toss_outcomes = [self.coin_toss(times=1, p_head=random.uniform(0, 1))[0] for i in range(n_times)]
            n_HT = 0
            n_TH = 0
            for i in range(n_times-1):
                if toss_outcomes[i] == 1 and toss_outcomes[i+1] == 0: # 1 = Head, 0 = Tail
                    n_HT += 1
                elif toss_outcomes[i] == 0 and toss_outcomes[i+1] == 1: # 1 = Head, 0 = Tail
                    n_TH += 1
            print(f"n_H counts = {toss_outcomes.count(0)}, n_T counts = {toss_outcomes.count(1)}, total counts = {len(toss_outcomes)}.")
            print(f"n_HT ({n_HT}) = n_TH±1 ({n_TH})? {abs(n_HT-n_TH) <= 1}")

    def fair_coin_toss_problem(self):
        """
        at least 3 different ways of tossing a fair coin
        """
        n_toss = int(1e5)
        for tosser in ['#1', '#2', '#3']:
            if tosser == '#1':
                toss_outcomes = self.fair_coin_toss(size=n_toss)
            elif tosser == '#2':
                toss_outcomes = self.coin_toss(times=n_toss, p_head=0.5)
            elif tosser == '#3':
                toss_outcomes = stats.binom.rvs(1, p=0.5, size=n_toss)
            print(f"\ntosser{tosser}:")
            n_head = (toss_outcomes == 1).sum()
            n_tail = (toss_outcomes == 0).sum()
            print(f"tossing a fair coin {n_toss} times: n_head = {n_head}, n_tail = {n_tail}")


def demo_probability():
    prob = probability()
    prob.plot_binomial()
    prob.head_tail_problem()
    prob.fair_coin_toss_problem()


def demo_distance():
    from ..datasets import public_dataset
    data = public_dataset(name="iris")
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    print("Minkowski distance of the 'sepal length(cm)' and 'sepal width(cm)' from the iris data:")
    for p_power in [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]:
        p = 2**p_power
        print(f"order p={p: .3f}, Minkowski distance = {distance(p=p).Minkowski(x=x, y=y): .3f}")
    print(f"Euclidean distance = {distance().Euclidean(x=x, y=y): .3f}")


def stats_demo():
    demo_probability()
    demo_distance()
