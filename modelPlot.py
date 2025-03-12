import numpy as np
import matplotlib.pyplot as plt


class ModelPlot:

    def __init__(self, check_every, total_rewards, title='Progress', filename:str = None, save=False, show=True):
        plt.style.use('seaborn-v0_8-dark') # Albo np. 'ggplot'

        self.draw_plot(check_every, total_rewards, title, filename, save, show)

    @classmethod
    def draw_plot(cls, check_every, total_rewards, title='Progress', filename:str = None, save=False, show=True):

        # Compute statistics
        avg_rewards = [np.mean(rewards) for rewards in total_rewards]
        max_rewards = [np.max(rewards) for rewards in total_rewards]
        min_rewards = [np.min(rewards) for rewards in total_rewards]
        epochs = list(range(check_every, len(total_rewards)*check_every+1, check_every))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, avg_rewards, label='Avg Reward', marker='o')
        plt.plot(epochs, max_rewards, label='Max Reward', marker='s')
        plt.plot(epochs, min_rewards, label='Min Reward', marker='d')

        plt.xlabel('Epochs')
        plt.ylabel('Reward')
        plt.title(title)
        plt.legend()

        # plt.grid(True)
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

        if save and filename:
            plt.savefig(filename, dpi=300)

        if show:
            plt.show()
