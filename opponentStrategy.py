from abc import ABC, abstractmethod
from typing import Type
import numpy as np


class Strategy(ABC):

    def __str__(self):
        return self.__class__.__name__

    @abstractmethod
    def get_action(self, history, probabilities):
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
            self.repeats = self.strategies_sequence[self.current_action][1]
        if self.current_action == 0 and self.shuffle:
            self.shuffle_strategies()
        return self.strategies_sequence[self.current_action][0]

    def get_action(self, agent_history, probabilities):
        strategy = self._next_action()
        return strategy.get_action(agent_history, probabilities)

    def add_strategy(self, strategy:Strategy | list[Strategy], repeats=None) -> Type['OpponentStrategy']:
        if isinstance(strategy, list):
            for s in strategy:
                self.add_strategy(s, repeats)
        else:
            self.strategies_sequence.append((strategy, repeats or self.repeats))
        return self

    def build_sequence(self, repeats=None):
        self.add_strategy([TitForTat(), TitForTatWithProbability(), RandomDecision(), Opposite()], repeats)
        return self

    def shuffle_strategies(self):
        np.random.shuffle(self.strategies_sequence)

    def _names(self, short=True):
        if short:
            sequence = [strategy.short_name for strategy, _ in self.strategies_sequence]
        else:
            sequence = [strategy.name for strategy, _ in self.strategies_sequence]
        if self.shuffle and isinstance(sequence, list):
            sequence = '_'.join(sorted(sequence))
        return sequence

    def short_names(self):
        return self._names(short=True)

    def names(self):
        return self._names(short=False)

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

    def get_action(self, history, *_) -> int:
        if len(history) > 0:
            # Tit-for-tat: repeat agent's previous action.
            return history[-1][0]
        else:
            return 0  # cooperate the first round


class TitForTatWithProbability(Strategy):

    def __init__(self, probability=0.8):
        self.probability = probability

    def get_action(self, history, *_) -> int:
        if len(history) > 0:
            action = np.random.choice([history[-1][0], 1 - history[-1][0]], p=[self.probability, 1 - self.probability])
        else:
            action = np.random.choice([0, 1])
        return action


class RandomDecision(Strategy):

    def get_action(self, *_) -> int:
        return np.random.choice([0, 1])


class Opposite(Strategy):

    def get_action(self, history, *_) -> int:
        if len(history) > 0:
            action = 1 - history[-1][0]
        else:
            action = np.random.choice([0, 1])
        return action
