from model import Model, initialize_random_seed
from opponentStrategy import OpponentStrategy, tit_for_tat, tit_for_tat_with_probability


# Set random seeds for reproducibility
SEED = 420

# Game configuration
NUM_ROUNDS_RANGE = (20, 100)
NUM_EPISODES = 200  # number of training episodes
DISCOUNT_FACTOR = 0.9

# OPPONENT_STRATEGY = OpponentStrategy().go_infinity
# OPPONENT_STRATEGY = tit_for_tat
OPPONENT_STRATEGY = tit_for_tat_with_probability

strategy_name = (n:=OPPONENT_STRATEGY.__name__.replace('_', ''))[:3] + n[-3:]
num_rounds_str = f'{NUM_ROUNDS_RANGE[0]}_{NUM_ROUNDS_RANGE[1]}'
MODEL_PATH = f'{Model.__module__}'

MODEL_FILENAME = f'{strategy_name}-{NUM_EPISODES}-{num_rounds_str}-{DISCOUNT_FACTOR}.h5'
MODEL_PATH = f'{MODEL_PATH}/{MODEL_FILENAME}'

def main():
    model = Model(MODEL_PATH, NUM_ROUNDS_RANGE, NUM_EPISODES,OPPONENT_STRATEGY, DISCOUNT_FACTOR, SEED)
    if not model.load_model():
        model.train_agent()
    model.run_game(60)


if __name__ == '__main__':
    main()
