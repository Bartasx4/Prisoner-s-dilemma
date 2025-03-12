from abc import ABC, abstractmethod
import numpy as np


class Strategy(ABC):
    """Abstract base class for defining strategies."""

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get_action(self, history, probabilities):
        """Defines the action selection logic for the strategy."""
        pass

    @property
    def name(self) -> str:
        """Full name of the strategy."""
        return self.__class__.__name__

    @property
    def short_name(self):
        if len(self.__class__.__name__) <= 6:
            return self.__class__.__name__
        return (n:=self.__class__.__name__)[:3] + '-' + n[-3:]


class OpponentStrategy:
    """
    Manages opponent strategies in a game.

    This class allows dynamic switching, shuffling, and sequencing
    of multiple strategies for the opponent in iterative games.
    """

    def __init__(self, repeats: int = 7, shuffle: bool = True):
        """
        Initialize the strategy manager.
        """
        self.repeats = repeats
        self.counter = -1
        self.current_action = 0
        self.shuffle = shuffle
        self.strategies_sequence: list[tuple[Strategy, int]] = []

    def _next_action(self) -> Strategy:
        """Switches to the next strategy in the sequence."""
        self.counter += 1
        if self.counter == self.repeats:
            self.counter = 0
            self.current_action = (self.current_action + 1) % len(self.strategies_sequence)
            self.repeats = self.strategies_sequence[self.current_action][1]
        if self.current_action == 0 and self.shuffle:
            self.shuffle_strategies()
        return self.strategies_sequence[self.current_action][0]

    def get_action(self, agent_history, probabilities) -> int:
        """Gets the next action from the currently active strategy."""
        strategy = self._next_action()
        return strategy.get_action(agent_history, probabilities)

    def add_strategy(self, strategy:Strategy | list[Strategy], repeats=None) -> 'OpponentStrategy':
        if isinstance(strategy, list):
            for s in strategy:
                self.add_strategy(s, repeats)
        else:
            self.strategies_sequence.append((strategy, repeats or self.repeats))
        return self

    def build_sequence(self, repeats: int = None) -> 'OpponentStrategy':
        """
        Pre-configures a default sequence of strategies.
        """
        self.add_strategy([TitForTat(), TitForTatWithProbability(), RandomDecision(), Opposite()], repeats)
        return self

    def shuffle_strategies(self):
        """Randomly shuffles the sequence of strategies."""
        np.random.shuffle(self.strategies_sequence)

    def _names(self, short=True, connector='_') -> str:
        """
                Helper function to generate a sequence of strategy names.
        """
        sequence = [strategy.short_name if short else strategy.name for strategy, _ in self.strategies_sequence]
        if self.shuffle:
            sequence = sorted(sequence)  # Sort only for display purposes
        return connector.join(sequence)

    def short_names(self) -> str:
        """Returns a joined string of short names for strategies."""
        return self._names(short=True)

    def clean_names(self) -> str:
        """Returns a joined string of full names for strategies."""
        return self._names(short=False, connector=', ')

    def names(self) -> str:
        """Alias for clean_names."""
        return self.clean_names()

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


# Concrete strategy classes

class TitForTat(Strategy):
    """Repeats the opponent's previous action (if available)."""

    def get_action(self, history, *_) -> int:
        if len(history) > 0:
            # Tit-for-tat: repeat agent's previous action.
            return history[-1][0]
        else:
            return 0  # cooperate the first round


class TitForTatWithProbability(Strategy):
    """Tit-for-Tat strategy with probabilistic deviations."""

    def __init__(self, probability=0.8):
        self.probability = probability

    def get_action(self, history, *_) -> int:
        if len(history) > 0:
            action = np.random.choice([history[-1][0], 1 - history[-1][0]], p=[self.probability, 1 - self.probability])
        else:
            action = np.random.choice([0, 1])
        return action


class RandomDecision(Strategy):
    """Always selects a random action."""

    def get_action(self, *_) -> int:
        return np.random.choice([0, 1])


class Opposite(Strategy):
    """Always opposes the opponent's previous action."""

    def get_action(self, history, *_) -> int:
        if len(history) > 0:
            action = 1 - history[-1][0]
        else:
            action = np.random.choice([0, 1])
        return action
