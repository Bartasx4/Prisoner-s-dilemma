import numpy as np
import pandas as pd
import torch

from opponentStrategy import OpponentStrategy, Strategy


# Payoff matrix for the agent (row) and opponent (column)
PAYOFFS = {
    (0, 0): 3,  # Both cooperate
    (0, 1): 0,  # Agent cooperates, opponent defects
    (1, 0): 5,  # Agent defects, opponent cooperates
    (1, 1): 1,  # Both defect
}

ACTION_SPACE = [0, 1]  # Action space: 0 = Cooperate, 1 = Defect


class Game:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def calculate_points(history: list[tuple[int, int]]):
        agent_points = [PAYOFFS[(agent_action, opponent_action)] for agent_action, opponent_action in history]
        opponent_points = [PAYOFFS[(opponent_action, agent_action)] for agent_action, opponent_action in history]
        return agent_points, opponent_points

    @staticmethod
    def play_episode(agent_strategy, num_rounds: int | tuple[int, int], opponent_strategy: OpponentStrategy | Strategy,
                     training=True):
        """
        Simulates a single episode of the game between the agent and the opponent.
        Returns episode states, actions, and rewards if training, otherwise the history.
        """
        history = []
        episode_states, episode_actions, agent_rewards = [], [], []
        num_rounds = get_rounds_number(num_rounds)
        for t in range(num_rounds):
            agent_action = int(agent_strategy.get_action(history, num_rounds))
            opponent_action = int(opponent_strategy.get_action(history, num_rounds))

            episode_actions.append(agent_action)
            episode_states.append(get_state(history, num_rounds))

            agent_reward = PAYOFFS[(agent_action, opponent_action)]
            agent_rewards.append(agent_reward)
            history.append((agent_action, opponent_action))

        if training:
            return episode_states, episode_actions, agent_rewards
        else:
            return history

    def run_game(self, agent_strategy, num_rounds: int | tuple[int, int],
                 opponent_strategy: OpponentStrategy | Strategy, display_max=50):
        """
        Runs and displays details of a single game between the agent and the opponent.
        """
        history = self.play_episode(agent_strategy, num_rounds, opponent_strategy, training=False)

        agent_moves = ['Cooperate' if a == 0 else 'Defect' for a, _ in history]
        opponent_moves = ['Cooperate' if o == 0 else 'Defect' for _, o in history]
        agent_points, opponent_points = self.calculate_points(history)

        rounds = list(range(1, len(agent_moves) + 1))
        df = pd.DataFrame({
            'Round': rounds,
            'Agent Move': agent_moves,
            'Opponent Move': opponent_moves,
            'Agent Points': agent_points,
            'Opponent Points': opponent_points,
        })

        if len(df) > display_max:
            df = pd.concat([df.head(display_max // 2), df.tail(display_max // 2)])

        print('\nGame play-by-play:')
        print(df.to_string(index=False))
        print(f'\nFinal total reward, agent vs opponent: {sum(agent_points)} vs {sum(opponent_points)}')

    def run_multiple_games(self, agent_strategy, num_rounds: int | tuple[int, int],
                           opponent_strategy: OpponentStrategy | Strategy, num_games=10):
        """
        Simulates multiple games and calculates average rewards for both the agent and the opponent.
        """
        total_agent_rewards, total_opponent_rewards = [], []

        for _ in range(num_games):
            history = self.play_episode(agent_strategy, num_rounds, opponent_strategy, training=False)
            agent_points, opponent_points = self.calculate_points(history)

            total_agent_rewards.append(np.sum(agent_points))
            total_opponent_rewards.append(np.sum(opponent_points))

        print(
            f'\nAverage total rewards: \nAgent: {np.mean(total_agent_rewards):.2f} \nOpponent: {np.mean(total_opponent_rewards):.2f}')


def get_rounds_number(num_rounds: int | tuple[int, int]) -> int:
    """
    Returns the number of rounds, either as a fixed value or sampled from a range.
    """
    return num_rounds if isinstance(num_rounds, int) else np.random.randint(num_rounds[0], num_rounds[1])


def get_state(history: list[tuple[int, int]], num_rounds: int) -> np.ndarray:
    """
    Returns a padded representation of the game's history.
    """
    state_seq = [np.array([agent, opponent]) for agent, opponent in history]
    padded = np.zeros((num_rounds, 2), dtype=np.float32)

    if state_seq:
        padded[:len(state_seq), :] = np.array(state_seq, dtype=np.float32)

    return padded
