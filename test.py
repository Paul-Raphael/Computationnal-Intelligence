from random import random
from math import ceil
from functools import reduce
from collections import namedtuple, deque
from queue import PriorityQueue

import numpy as np
import copy
from tqdm.auto import tqdm

State = namedtuple('State', ['taken', 'not_taken'])


def covered(state):
    return reduce(
        np.logical_or,
        [SETS[i] for i in state.taken],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    )


def goal_check(state):
    return np.all(covered(state))

PROBLEM_SIZE = 30
NUM_SETS = 80
SETS = tuple(np.array([random() < 0.1 for _ in range(PROBLEM_SIZE)]) for _ in range(NUM_SETS))
assert goal_check(State(set(range(NUM_SETS)), set())), "Probelm not solvable"

def h9(state):
    updated_state = copy.deepcopy(state)
    already_covered = covered(updated_state)  # already covered elements
    if np.all(already_covered):
        return 0

    missing_size = PROBLEM_SIZE - np.sum(already_covered)  # number of elements not covered

    candidates = sorted(updated_state.not_taken, key=lambda x: np.sum(SETS[x] & ~already_covered), reverse=True)

    taken = 0
    while missing_size > 0:
        if taken >= len(candidates):
            return float('inf')  # No more candidates, cannot cover the missing elements

        selected_candidate = candidates[taken]
        updated_state.taken.add(selected_candidate)
        updated_state.not_taken.remove(selected_candidate)
        already_covered |= SETS[selected_candidate]
        missing_size = PROBLEM_SIZE - np.sum(already_covered)
        taken += 1

    return taken


def f(state):
    return len(state.taken) + h9(state)

frontier = PriorityQueue()
state = State(set(), set(range(NUM_SETS)))
frontier.put((f(state), state))

counter = 0
_, current_state = frontier.get()
with tqdm(total=None) as pbar:
    while not goal_check(current_state):
        counter += 1
        for action in current_state[1]:
            new_state = State(
                current_state.taken ^ {action},
                current_state.not_taken ^ {action},
            )
            frontier.put((f(new_state), new_state))
            #print(f(new_state), new_state)

        _, current_state = frontier.get()
        pbar.update(1)

print(f"Solved in {counter:,} steps ({len(current_state.taken)} tiles)")