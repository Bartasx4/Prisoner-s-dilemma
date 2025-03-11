from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from game import Game, get_rounds_number
from modelPlot import ModelPlot

# Payoff matrix for the agent (row) and opponent (column)
PAYOFFS = {
    (0, 0): 3,  # Both cooperate
    (0, 1): 0,  # Agent cooperates, opponent defects
    (1, 0): 5,  # Agent defects, opponent cooperates
    (1, 1): 1,  # Both defect
}

ACTION_SPACE = [0, 1]  # Action space: 0 = Cooperate, 1 = Defect
INPUT_SIZE = 2  # Model input size: [agent, opponent]

DISCOUNT_FACTOR = 0.9
EVAL_EVERY = 10


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

    def __init__(self, num_rounds_range: int | tuple[int, int], num_episodes: int, opponent_strategy, seed: int | None = None):
        self.num_rounds_range = num_rounds_range
        self.num_episodes = num_episodes
        self.opponent_strategy = opponent_strategy
        self.discount_factor = DISCOUNT_FACTOR
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

    def create_file_name(self, short=True) -> str:
        return f'{self.create_model_name(short)}.h5'

    def create_model_name(self, short=True) -> str:
        num_rounds_str = f'{self.num_rounds_range[0]}_{self.num_rounds_range[1]}'
        names = self.opponent_strategy.short_names() if short else self.opponent_strategy.names()
        return f'{self.num_episodes}-{num_rounds_str}-{names}'

    def get_action(self, history, num_rounds):
        state = get_state(history, num_rounds)
        state_input = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probabilities = self.model(state_input).detach().cpu().numpy()[0]
        action = np.random.choice(ACTION_SPACE, p=action_probabilities)
        return action, action_probabilities, state

    def train_agent(self):
        self.model = self.build_policy_network()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        eval_histories = []

        for ep in range(self.num_episodes):
            if ep % EVAL_EVERY == 0 and ep != 0:
                eval_histories.append(self.eval_policy(40))

            num_rounds = get_rounds_number(self.num_rounds_range)
            episode_states, episode_actions, episode_rewards = Game().play_episode(self,
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

        self.model.eval()
        if self.save_model():
            print(f"Training complete. Model saved to '{self.path}'.")
        title = self.create_model_name(short=False)
        ModelPlot(check_every=EVAL_EVERY,
                  total_rewards=eval_histories,
                  title=title,
                  filename=f'{self.__module__}/{title}.png',
                  save=True,
                  show=True)
        return self.model

    def save_model(self, overwrite=False) -> bool:
        path = Path(self.path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or overwrite:
            torch.save(self.model.state_dict(), self.path)
            return True
        return False

    def load_model(self):
        model_filename = self.path
        if Path(model_filename).exists():
            self.model = self.build_policy_network()
            print(f'Loading model from file: "{model_filename}"')
            self.model.load_state_dict(torch.load(model_filename))
            self.model.eval()
            return self.model
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
                history, rewards = Game().play_episode(self, num_rounds, self.opponent_strategy, training=False)
                total_reward = np.sum(rewards)
                total_rewards.append(total_reward)
                all_histories.append(history)
        return total_rewards

    @property
    def path(self):
        return f'{self.__module__}/{self.create_file_name()}'


def get_state(history: list[tuple[int, int]], num_rounds: int) -> np.ndarray:
    state_seq = []
    for (agent, opponent) in history:
        state_seq.append(np.array([agent, opponent]))
    padded = np.zeros((num_rounds, 2), dtype=np.float32)
    if len(state_seq) > 0:
        padded[:len(state_seq), :] = np.array(state_seq, dtype=np.float32)
    return padded
