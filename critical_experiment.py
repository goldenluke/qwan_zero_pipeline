import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

from qwan_zero_pipeline import (
    Net,
    ReplayBuffer,
    self_play,
    train,
    elo_from_score
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriticalField:
    def __init__(self):
        self.xi = 1.0
    def update(self):
        self.xi = np.random.uniform(0.5, 3.0)

def train_group(args, use_critical=False):

    net = Net().to(DEVICE)
    buffer = ReplayBuffer(args.buffer)

    elo_history = []

    cf = CriticalField()

    for gen in range(args.generations):

        if use_critical:
            cf.update()

        self_play(net, buffer, args.games, args.sims)
        train(net, buffer, args.batches, args.batch_size)

        score = np.random.uniform(0.3,0.7)  # placeholder arena
        elo = elo_from_score(score, 1200)
        elo_history.append(elo)

    return elo_history

def pipeline(args):

    print("Training Control...")
    elo_A = train_group(args, False)

    print("Training Critical...")
    elo_B = train_group(args, True)

    t_stat, p_value = stats.ttest_ind(elo_A, elo_B)

    print("Mean Elo Control:", np.mean(elo_A))
    print("Mean Elo Critical:", np.mean(elo_B))
    print("p-value:", p_value)

    plt.plot(elo_A,label="Control")
    plt.plot(elo_B,label="Critical")
    plt.legend()
    plt.savefig("elo_comparison.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",type=int,default=3)
    parser.add_argument("--games",type=int,default=50)
    parser.add_argument("--sims",type=int,default=30)
    parser.add_argument("--batches",type=int,default=10)
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--buffer",type=int,default=50000)

    args=parser.parse_args()
    pipeline(args)
