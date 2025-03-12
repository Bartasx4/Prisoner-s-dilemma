import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from game import Game, get_rounds_number, get_state
from policyNetwork import LSTMSoftmaxReluPolicyNetwork
from modelPlot import ModelPlot

EVAL_INFO_FILENAME = 'eval_info.txt'

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
EVAL_EPISODES = 10
NUM_ROUNDS_RANGE = 40


def initialize_random_seed(seed: int | None):
    """Initializes randomness using the provided seed."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


class Model:

    def __init__(self, num_rounds_range: int | tuple[int, int], num_episodes: int,
                 opponent_strategy, seed: int | None = None, policy_network=None):
        self.num_rounds_range = num_rounds_range
        self.num_episodes = num_episodes
        self.opponent_strategy = opponent_strategy
        self.discount_factor = DISCOUNT_FACTOR
        self.policy_network = policy_network or LSTMSoftmaxReluPolicyNetwork()
        self.seed = seed
        self.model = None
        self.eval_every = EVAL_EVERY
        self.eval_episodes = EVAL_EPISODES
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        initialize_random_seed(self.seed)

    def build_policy_network(self):
        """Creates and returns a new instance of the policy network."""
        return self.policy_network.__class__().to(self.device)

    def compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        """Calculates discounted returns for the rewards."""
        returns = np.zeros_like(rewards, dtype=np.float32)
        cumulative = 0.0
        for t in reversed(range(len(rewards))):
            cumulative = rewards[t] + self.discount_factor * cumulative
            returns[t] = cumulative
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns

    def create_file_name(self, short=True) -> str:
        """Generates a file name for the model."""
        return f'{self.create_model_name(short)}.h5'

    def create_model_name(self, short=True) -> str:
        """Generates a model name based on configuration."""
        num_rounds_str = f'{self.num_rounds_range[0]}_{self.num_rounds_range[1]}'
        names = self.opponent_strategy.short_names() if short else self.opponent_strategy.names()
        return f'{self.num_episodes}-{num_rounds_str}-{names}'

    def get_action(self, history, num_rounds) -> int:
        """Determines the next action based on the current state."""
        state = get_state(history, num_rounds)
        state_input = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probabilities = self.model(state_input).detach().cpu().numpy()[0]
        action = np.random.choice(ACTION_SPACE, p=action_probabilities)
        return int(action)

    def run_training_game(self, optimizer):
        """Executes a single training game and updates the model."""
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

        return episode_states, episode_actions, episode_rewards, loss

    def train_agent(self, save_eval_data=True):
        """Trains the agent over multiple episodes."""
        self.model = self.build_policy_network()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        eval_histories = []

        for episode in range(self.num_episodes):
            if episode % self.eval_every == 0 and episode != 0:
                eval_history = self.eval_policy(self.eval_episodes, NUM_ROUNDS_RANGE)
                eval_rewards = np.sum(eval_history, axis=1)
                eval_histories.append(eval_rewards)

            episode_states, episode_actions, episode_rewards, loss = self.run_training_game(optimizer)
            total_reward = np.sum(episode_rewards)
            print(f'Episode {episode + 1}/{self.num_episodes}: Loss = {loss.item():.3f}, Total Reward = {total_reward}')

        eval_histories = np.array(eval_histories)
        self.model.eval()

        if self.save_model():
            print(f"Training complete. Model saved to '{self.path}'.")

        if save_eval_data:
            self._save_eval_data(eval_histories)

        title = self.create_model_name(short=False)
        agent_rewards = [reward for reward, _ in np.sum(eval_histories, axis=1)]
        ModelPlot(check_every=self.eval_every,
                  total_rewards=agent_rewards,
                  title=title,
                  filename=f'{self.__module__}/{title}.png',
                  save=True,
                  show=True)
        return self.model

    def _save_eval_data(self, eval_histories: np.array = None):
        """Saves evaluation histories and model configuration to a file."""
        info = self.policy_network.info
        info.extend([
            ('num_rounds_range', self.num_rounds_range),
            ('num_episodes', self.num_episodes),
            ('opponent_strategy', self.opponent_strategy.clean_names()),
            ('discount_factor', self.discount_factor),
            ('seed', self.seed),
            ('eval_every', self.eval_every),
            ('eval_episodes', self.eval_episodes),
            ('eval_histories', eval_histories.tolist())
        ])
        info_str = json.dumps(dict(info), indent=4)
        file_path = f'{self.__module__}/{EVAL_INFO_FILENAME}'
        with open(file_path, 'a', encoding='utf-8') as file:
            if file.tell() > 0:
                info_str = ',\n' + info_str
            file.write(info_str + '\n')

    def save_model(self, overwrite=False) -> bool:
        """Saves the model to disk."""
        path = Path(self.path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or overwrite:
            torch.save(self.model.state_dict(), self.path)
            return True
        return False

    def set_evaluation(self, eval_every: int = None, eval_episodes: int = None):
        """Sets evaluation parameters."""
        self.eval_every = eval_every or self.eval_every
        self.eval_episodes = eval_episodes or self.eval_episodes

    def load_model(self):
        """Loads a model from disk if it exists."""
        model_filename = self.path
        if Path(model_filename).exists():
            self.model = self.build_policy_network()
            print(f'Loading model from file: "{model_filename}"')
            self.model.load_state_dict(torch.load(model_filename))
            self.model.eval()
            return self.model
        return None

    def eval_policy(self, num_eval_episodes: int = 10, num_rounds_range=None):
        """Evaluates the policy over multiple episodes."""
        if self.model is None:
            raise ValueError(
                "The model has not been trained or loaded. Call 'train_agent' or 'load_model' first.")
        all_histories = []
        num_rounds_range = num_rounds_range or self.num_rounds_range
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations as this is not training
            for _ in range(num_eval_episodes):
                num_rounds = get_rounds_number(num_rounds_range)
                history = Game().play_episode(self, num_rounds, self.opponent_strategy, training=False)
                all_histories.append(history)
        return all_histories

    @property
    def path(self):
        """Returns the path to save/load the model."""
        return f'{self.__module__}/{self.create_file_name()}'
