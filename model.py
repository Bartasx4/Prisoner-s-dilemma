from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from game import Game, get_rounds_number

# Payoff matrix for the agent (row) and opponent (column)
PAYOFFS = {
    (0, 0): 3,  # Both cooperate
    (0, 1): 0,  # Agent cooperates, opponent defects
    (1, 0): 5,  # Agent defects, opponent cooperates
    (1, 1): 1,  # Both defect
}

ACTION_SPACE = [0, 1]  # Action space: 0 = Cooperate, 1 = Defect
INPUT_SIZE = 2  # Model input size: [agent, opponent]


def initialize_random_seed(seed: int | None):
    """Initializes randomness using the seed."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=16, batch_first=True)
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        return torch.softmax(self.fc2(x), dim=-1)


class Model:

    def __init__(self, model_filename: str, num_rounds_range: int | tuple[int, int], num_episodes: int, opponent_strategy,
                 discount_factor=0.9, seed: int | None = None):
        self.model_filename = model_filename
        self.num_rounds_range = num_rounds_range
        self.num_episodes = num_episodes
        self.opponent_strategy = opponent_strategy
        self.discount_factor = discount_factor
        self.seed = seed
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        initialize_random_seed(self.seed)

    def build_policy_network(self):
        return PolicyNetwork().to(self.device)

    def compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        returns = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0.
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.discount_factor * cumulative
            returns[t] = cumulative
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns

    def train_agent(self):
        self.model = self.build_policy_network()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for ep in range(self.num_episodes):
            if ep % 100 == 0 and ep != 0:
                self.eval_policy(30)

            num_rounds = get_rounds_number(self.num_rounds_range)
            episode_states, episode_actions, episode_rewards = Game().play_episode(self.model,
                                                                                   num_rounds,
                                                                                   self.opponent_strategy,
                                                                                   training=True)
            returns = self.compute_returns(episode_rewards)

            states_tensor = torch.tensor(episode_states, dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(episode_actions, dtype=torch.long, device=self.device)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

            optimizer.zero_grad()

            probs = self.model(states_tensor)
            m = Categorical(probs)
            action_probs = m.log_prob(actions_tensor)

            loss = -torch.mean(action_probs * returns_tensor)

            loss.backward()
            optimizer.step()

            total_reward = np.sum(episode_rewards)
            print(f'Episode {ep + 1}/{self.num_episodes}: Loss = {loss.item():.3f}, Total Reward = {total_reward}')

        self.save_model()
        print(f'Training complete. Model saved to \'{self.model_filename}\'.')
        return self.model

    def save_model(self, overwrite=False):
        path = Path(self.model_filename)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or overwrite:
            torch.save(self.model.state_dict(), self.model_filename)

    def load_model(self):
        if Path(self.model_filename).exists():
            model = self.build_policy_network()
            print(f'Loading model from file: "{self.model_filename}"')
            model.load_state_dict(torch.load(self.model_filename))
            model.eval()
            self.model = model
            return model
        return None

    def eval_policy(self, num_eval_episodes: int = 10):
        if self.model is None:
            raise ValueError(
                "The model has not been trained or loaded. Call 'train_agent' or 'load_model' first.")

        total_rewards = []
        all_histories = []
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations as this is not training
            for _ in range(num_eval_episodes):
                # Play an episode in non-training mode
                num_rounds = get_rounds_number(self.num_rounds_range)
                history, rewards = Game().play_episode(self.model,num_rounds,self.opponent_strategy, training=False)
                total_reward = np.sum(rewards)
                total_rewards.append(total_reward)
                all_histories.append(history)

        # Compute statistics
        avg_reward = np.mean(total_rewards)
        max_reward = np.max(total_rewards)
        min_reward = np.min(total_rewards)

        print(f'\nModel evaluation on {num_eval_episodes} episodes:')
        print(f'Average reward: {avg_reward:.2f}')
        print(f'Highest reward: {max_reward}')
        print(f'Lowest reward: {min_reward}')

        # Return evaluation details as a dictionary
        return dict(average_reward=avg_reward, max_reward=max_reward, min_reward=min_reward,
                    total_rewards_per_episode=total_rewards, histories=all_histories)
