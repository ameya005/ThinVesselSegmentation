import random

__author__ = 'kushal'
"""
Provides utilies and preporcessing facilities
"""


def compute_random(X, y, samples=10000, seed=42):
    for key in X.keys():
        rnumber = random.sample(xrange(len(X[key])), samples)
