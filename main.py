from model import Model, initialize_random_seed
from game import Game
from opponentStrategy import OpponentStrategy


# Set random seeds for reproducibility
SEED = 42

# Game configuration
NUM_ROUNDS_RANGE = (30, 100)
NUM_EPISODES = 100  # number of training episodes

OPPONENT_STRATEGY = OpponentStrategy().build_sequence()

num_rounds_str = f'{NUM_ROUNDS_RANGE[0]}_{NUM_ROUNDS_RANGE[1]}'
MODEL_PATH = f'{Model.__module__}'

MODEL_FILENAME = f'{NUM_EPISODES}-{num_rounds_str}-{str(OPPONENT_STRATEGY)}.h5'
MODEL_PATH = f'{MODEL_PATH}/{MODEL_FILENAME}'

def main():
    model = Model(MODEL_PATH, NUM_ROUNDS_RANGE, NUM_EPISODES,OPPONENT_STRATEGY,SEED)
    if not model.load_model():
        model.train_agent()
    Game().run_game(model.model, 60, OPPONENT_STRATEGY)


if __name__ == '__main__':
    main()
