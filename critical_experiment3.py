import argparse
import numpy as np
import torch
import chess
import chess.engine
from tqdm import tqdm

from qwan_zero_pipeline import (
    Net,
    ReplayBuffer,
    self_play,
    train,
    MCTS,
    DEVICE
)

# ============================================================
# CAMPO CRÍTICO
# ============================================================

class CriticalField:
    def __init__(self):
        self.xi = 1.0

    def update(self):
        self.xi = np.random.uniform(0.5, 3.0)

    def cpuct(self):
        return 1 + 2/(1+self.xi)

# ============================================================
# ARENA MCTS vs MCTS
# ============================================================

def play_mcts_vs_mcts(netA, netB, sims):

    board = chess.Board()
    mctsA = MCTS(netA, sims=sims)
    mctsB = MCTS(netB, sims=sims)

    while not board.is_game_over():
        if board.turn:
            pi = mctsA.search(board)
        else:
            pi = mctsB.search(board)

        moves = list(pi.keys())
        probs = np.array(list(pi.values()))
        move = np.random.choice(moves,p=probs)
        board.push(move)

    return board.result()

def arena(netA, netB, games, sims):

    winsA=winsB=draws=0

    for _ in tqdm(range(games), desc="Arena"):
        result = play_mcts_vs_mcts(netA, netB, sims)

        if result=="1-0": winsA+=1
        elif result=="0-1": winsB+=1
        else: draws+=1

    return winsA,winsB,draws

# ============================================================
# TREINAMENTO
# ============================================================

def train_group(args, use_critical=False):

    net = Net().to(DEVICE)
    buffer = ReplayBuffer(args.buffer)
    cf = CriticalField()

    for gen in range(args.generations):

        if use_critical:
            cf.update()
            c_puct = cf.cpuct()
            dirichlet = 0.3 + 0.5*np.exp(-cf.xi)
            temperature = 1/(1+cf.xi)
        else:
            c_puct = 1.5
            dirichlet = 0.3
            temperature = 1.0

        self_play(net, buffer, args.games, args.sims,
                  c_puct, dirichlet, temperature)

        train(net, buffer, args.batches, args.batch_size)

    return net

# ============================================================
# PIPELINE
# ============================================================

def main(args):

    print("Training Control...")
    net_control = train_group(args, False)

    print("Training Critical...")
    net_critical = train_group(args, True)

    wC,wK,d = arena(net_control, net_critical,
                    args.arena_games, args.sims)

    print("\n=== Control vs Critical Results ===")
    print("Control wins:", wC)
    print("Critical wins:", wK)
    print("Draws:", d)

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--generations",type=int,default=3)
    parser.add_argument("--games",type=int,default=50)
    parser.add_argument("--sims",type=int,default=50)
    parser.add_argument("--batches",type=int,default=20)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--buffer",type=int,default=50000)
    parser.add_argument("--arena_games",type=int,default=20)

    args = parser.parse_args()
    main(args)
