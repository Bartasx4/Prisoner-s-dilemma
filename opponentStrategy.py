from abc import ABC, abstractmethod
import numpy as np


class Strategy(ABC):

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_action(self, agent_history, probabilities):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def short_name(self):
        if len(self.__class__.__name__) <= 6:
            return self.__class__.__name__
        return (n:=self.__class__.__name__)[:3] + '-' + n[-3:]


class OpponentStrategy:

    def __init__(self, repeats=7, shuffle=True):
        self.repeats = repeats
        self.counter = -1
        self.current_action = 0
        self.shuffle = shuffle
        self.strategies_sequence: list[tuple[Strategy, int]] = []

    def _next_action(self):
        self.counter += 1
        if self.counter == self.repeats:
            self.counter = 0
            self.current_action = (self.current_action + 1) % len(self.strategies_sequence)
        if self.current_action == 0 and self.shuffle:
            self.shuffle_strategies()
        return self.strategies_sequence[self.current_action][0]

    def get_action(self, agent_history, probabilities):
        strategy = self._next_action()
        return strategy.get_action(agent_history, probabilities)

    def add_strategy(self, strategy:Strategy | list[Strategy], repeats=None):
        if isinstance(strategy, list):
            for s in strategy:
                self.add_strategy(s, repeats)
        else:
            self.strategies_sequence.append((strategy, repeats or self.repeats))

    def build_sequence(self, repeats=None):
        self.add_strategy([TitForTat(), TitForTatWithProbability(), RandomDecision(), RandomWithProbabilities(), Opposite()], repeats)
        return self

    def shuffle_strategies(self):
        np.random.shuffle(self.strategies_sequence)

    def __str__(self):
        sequence = [strategy.short_name for strategy, _ in self.strategies_sequence]
        if self.shuffle:
            sequence = sorted(sequence)
        return '_'.join(sequence)

    def __format__(self, format_spec):
        result = ''
        match format_spec:
            case '%r':
                result += f'{str(self.repeats)}'
            case '%s':
                result += f'{str(self.shuffle)}'
            case '%f':
                result += f'{str(self.strategies_sequence)}'
            case '-':
                result += '-'
            case _:
                raise ValueError


class TitForTat(Strategy):

    def get_action(self, agent_history, *args):
        if len(agent_history) > 0:
            # Tit-for-tat: repeat agent's previous action.
            return agent_history[-1]
        else:
            return 0  # cooperate the first round


class TitForTatWithProbability(Strategy):

    def __init__(self, probability=0.8):
        self.probability = probability

    def get_action(self, agent_history, *args):
        if len(agent_history) > 0:
            action = np.random.choice([agent_history[-1], 1 - agent_history[-1]], p=[self.probability, 1 - self.probability])
        else:
            action = np.random.choice([0, 1])
        return action


class RandomDecision(Strategy):

    def get_action(self, agent_history, probabilities):
        return np.random.choice([0, 1])


class RandomWithProbabilities(Strategy):

    def get_action(self, agent_history, probabilities):
        return np.random.choice([0, 1], p=probabilities)


class Opposite(Strategy):

    def get_action(self, agent_history, probabilities):
        if len(agent_history) > 0:
            action = 1 - agent_history[-1]
        else:
            action = np.random.choice([0, 1])
        return action
