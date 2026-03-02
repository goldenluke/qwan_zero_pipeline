import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
        return 1 + 2 / (1 + self.xi)

# ============================================================
# MCTS vs MCTS
# ============================================================

def play_mcts_vs_mcts(netA, netB, simsA, simsB):

    board = chess.Board()
    mctsA = MCTS(netA, sims=simsA)
    mctsB = MCTS(netB, sims=simsB)

    while not board.is_game_over():

        if board.turn:
            pi = mctsA.search(board)
        else:
            pi = mctsB.search(board)

        moves = list(pi.keys())
        probs = np.array(list(pi.values()))
        move = np.random.choice(moves, p=probs)

        board.push(move)

    return board.result()

def arena_mcts(netA, netB, games, simsA, simsB):

    winsA = 0
    winsB = 0
    draws = 0

    for _ in tqdm(range(games), desc="Arena MCTS vs MCTS"):
        result = play_mcts_vs_mcts(netA, netB, simsA, simsB)

        if result == "1-0":
            winsA += 1
        elif result == "0-1":
            winsB += 1
        else:
            draws += 1

    return winsA, winsB, draws

# ============================================================
# VS STOCKFISH
# ============================================================

def play_vs_stockfish(net, engine, depth, sims):

    board = chess.Board()
    mcts = MCTS(net, sims=sims)

    while not board.is_game_over():

        if board.turn:
            pi = mcts.search(board)
            moves = list(pi.keys())
            probs = np.array(list(pi.values()))
            move = np.random.choice(moves, p=probs)
        else:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            move = result.move

        board.push(move)

    return board.result()

def arena_stockfish(net, stockfish_path, depth, games, sims):

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    wins = 0
    losses = 0
    draws = 0

    for _ in tqdm(range(games), desc="Arena vs Stockfish"):

        result = play_vs_stockfish(net, engine, depth, sims)

        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1

    engine.quit()
    return wins, losses, draws

# ============================================================
# ELO
# ============================================================

def estimate_elo(wins, losses, draws, opponent_elo):

    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total

    if score <= 0:
        return opponent_elo - 800
    if score >= 1:
        return opponent_elo + 800

    return opponent_elo - 400 * np.log10(1/score - 1)

# ============================================================
# TREINAMENTO
# ============================================================

def train_group(args, use_critical=False):

    net = Net().to(DEVICE)
    buffer = ReplayBuffer(args.buffer)

    cf = CriticalField()

    for gen in range(args.generations):

        print(f"\nGeneration {gen} | Critical={use_critical}")

        if use_critical:
            cf.update()

        self_play(net, buffer, args.games, args.sims)
        train(net, buffer, args.batches, args.batch_size)

    return net

# ============================================================
# PIPELINE
# ============================================================

def pipeline(args):

    print("\n=== Training Control ===")
    net_control = train_group(args, False)

    print("\n=== Training Critical ===")
    net_critical = train_group(args, True)

    # -----------------------------
    # MCTS vs MCTS Arena
    # -----------------------------

    wC, wK, d = arena_mcts(
        net_control,
        net_critical,
        args.arena_games,
        args.sims,
        args.sims
    )

    print("\n=== Control vs Critical Results ===")
    print(f"Control wins: {wC}")
    print(f"Critical wins: {wK}")
    print(f"Draws: {d}")

    elo_diff = estimate_elo(wK, wC, d, 1500)
    print(f"Estimated Critical Elo vs Control: {elo_diff:.1f}")

    # -----------------------------
    # Stockfish Evaluation
    # -----------------------------

    if args.stockfish:

        print("\n=== Critical vs Stockfish ===")

        w, l, d = arena_stockfish(
            net_critical,
            args.stockfish,
            args.depth,
            args.arena_games,
            args.sims
        )

        elo = estimate_elo(w, l, d, args.stockfish_elo)

        print(f"Wins: {w}")
        print(f"Losses: {l}")
        print(f"Draws: {d}")
        print(f"Estimated Elo vs Stockfish: {elo:.1f}")

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer", type=int, default=50000)

    parser.add_argument("--arena_games", type=int, default=10)

    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--stockfish_elo", type=int, default=2500)

    args = parser.parse_args()

    pipeline(args)
