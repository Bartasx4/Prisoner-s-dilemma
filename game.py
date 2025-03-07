import numpy as np
import pandas as pd
import torch


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

    def play_episode(self, model, num_rounds: int | tuple[int, int], opponent_strategy, training=True):
        history = []
        episode_states = []
        episode_actions = []
        episode_rewards = []

        agent_history = []

        num_rounds = get_rounds_number(num_rounds)

        for t in range(num_rounds):
            state = get_state(history, num_rounds)
            episode_states.append(state)

            state_input = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_probabilities = model(state_input).detach().cpu().numpy()[0]
            agent_action = np.random.choice(ACTION_SPACE, p=action_probabilities)
            episode_actions.append(agent_action)

            opp_action = opponent_strategy(t, agent_history, action_probabilities)

            reward = PAYOFFS[(agent_action, opp_action)]
            episode_rewards.append(reward)

            history.append((agent_action, opp_action))
            agent_history.append(agent_action)

        if training:
            return episode_states, episode_actions, episode_rewards
        else:
            return history, episode_rewards

    def run_game(self, model, num_rounds: int | tuple[int, int], opponent_strategy):
        history, rewards = self.play_episode(model, num_rounds, opponent_strategy, training=False)
        agent_moves = ['Cooperate' if a == 0 else "Defect" for a, _ in history]
        opp_moves = ['Cooperate' if o == 0 else "Defect" for _, o in history]
        rounds = list(range(1, len(agent_moves) + 1))
        round_rewards = rewards
        cum_reward = np.sum(round_rewards)
        max_reward = [max([PAYOFFS[0, move], PAYOFFS[1, move]]) for _, move in history]
        min_reward = [min([PAYOFFS[0, move], PAYOFFS[1, move]]) for _, move in history]
        pd.set_option('display.max_rows', 50)
        df = pd.DataFrame({
            'Round': rounds,
            'Agent Move': agent_moves,
            'Opponent Move': opp_moves,
            'Reward': round_rewards,
            'Max Reward': max_reward,
            'Min Reward': min_reward,
        })
        if len(df) > 40:
            df = pd.concat([df.head(20), df.tail(20)])

        print('\nGame play-by-play:')
        print(df.to_string(index=False))
        print(f'\nFinal total reward for the agent: {sum(min_reward)} <= {cum_reward} <= {sum(max_reward)}')


def get_rounds_number(num_rounds: int | tuple[int, int]) -> int:
    return num_rounds if isinstance(num_rounds, int) else np.random.randint(num_rounds[0], num_rounds[1])


def get_state(history: list[tuple[int, int]], num_rounds: int) -> np.ndarray:
    state_seq = []
    for (agent, opponent) in history:
        state_seq.append(np.array([agent, opponent]))
    padded = np.zeros((num_rounds, 2), dtype=np.float32)
    if len(state_seq) > 0:
        padded[:len(state_seq), :] = np.array(state_seq, dtype=np.float32)
    return padded
