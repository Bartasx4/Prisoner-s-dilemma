import numpy as np


class OpponentStrategy:

    def __init__(self, repeats=7, shuffle=True, sequence=None):
        self.repeats = repeats
        self.counter = -1
        self.current_action = 0
        self.shuffle = shuffle
        self.sequence = sequence or [tit_for_tat, random_decision, random_with_probabilities, opposite, tit_for_tat_with_probability]
        if shuffle:
            np.random.shuffle(self.sequence)

    def go_infinity(self, round_idx, agent_history, probabilities):
        self.counter += 1
        if self.counter == self.repeats:
            self.counter = 0
            self.current_action = (self.current_action + 1) % len(self.sequence)
            if self.current_action == 0 and self.shuffle:
                np.random.shuffle(self.sequence)
        return self.sequence[self.current_action](round_idx, agent_history, probabilities)


def tit_for_tat(round_idx, agent_history, *args):
    if round_idx > 0:
        # Tit-for-tat: repeat agent's previous action.
        return agent_history[-1]
    else:
        return 0  # cooperate the first round


def tit_for_tat_with_probability(round_idx, agent_history, *args):
    prob = 0.8
    if round_idx > 0:
        action = np.random.choice([agent_history[-1], 1 - agent_history[-1] ], p=[prob, 1 - prob])
    else:
        action = np.random.choice([0, 1])
    return action


def opposite(round_idx, agent_history, *args):
    if round_idx > 0:
        action = 1 - agent_history[-1]
    else:
        action = np.random.choice([0, 1])
    return action


def random_decision(*args):
    action = np.random.choice([0, 1])
    return action


def random_with_probabilities(_, __, probabilities):
    action = np.random.choice([0, 1], p=probabilities)
    return action
