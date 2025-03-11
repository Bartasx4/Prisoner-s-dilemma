from model import Model, initialize_random_seed
from game import Game
from opponentStrategy import OpponentStrategy, RandomDecision


# Set random seeds for reproducibility
SEED = 1337

# Game configuration
NUM_ROUNDS_RANGE = (30, 100)
NUM_EPISODES = 200  # number of training episodes

OPPONENT_STRATEGY = OpponentStrategy().build_sequence()
# OPPONENT_STRATEGY = OpponentStrategy().add_strategy(RandomDecision())

def main():
    model = Model(NUM_ROUNDS_RANGE, NUM_EPISODES,OPPONENT_STRATEGY,SEED)
    if not model.load_model():
        model.train_agent()
    Game().run_game(model, 60, OPPONENT_STRATEGY)


if __name__ == '__main__':
    main()
