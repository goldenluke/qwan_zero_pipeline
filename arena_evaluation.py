import argparse
import chess
import chess.engine
import numpy as np
import torch
from tqdm import tqdm

from qwan_zero_pipeline import (
    Net,
    MCTS,
    board_to_tensor,
    move_to_index,
    DEVICE
)

# ============================================================
# PLAY MCTS vs MCTS
# ============================================================

def play_mcts_vs_mcts(netA, netB, simsA=50, simsB=50):

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

# ============================================================
# ARENA MCTS vs MCTS
# ============================================================

def arena_mcts(netA, netB, games, simsA, simsB):

    winsA = 0
    winsB = 0
    draws = 0

    for i in tqdm(range(games), desc="MCTS vs MCTS"):

        result = play_mcts_vs_mcts(netA, netB, simsA, simsB)

        if result == "1-0":
            winsA += 1
        elif result == "0-1":
            winsB += 1
        else:
            draws += 1

    return winsA, winsB, draws


# ============================================================
# PLAY vs STOCKFISH
# ============================================================

def play_vs_stockfish(net, engine, depth, sims):

    board = chess.Board()
    mcts = MCTS(net, sims=sims)

    while not board.is_game_over():

        if board.turn:
            # QWAN move
            pi = mcts.search(board)
            moves = list(pi.keys())
            probs = np.array(list(pi.values()))
            move = np.random.choice(moves, p=probs)
        else:
            # Stockfish move
            result = engine.play(board, chess.engine.Limit(depth=depth))
            move = result.move

        board.push(move)

    return board.result()


# ============================================================
# ARENA vs STOCKFISH
# ============================================================

def arena_stockfish(net, stockfish_path, depth, games, sims):

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    wins = 0
    losses = 0
    draws = 0

    for i in tqdm(range(games), desc="vs Stockfish"):

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
# ELO ESTIMATION
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
# CLI
# ============================================================

def main(args):

    netA = Net().to(DEVICE)
    netB = Net().to(DEVICE)

    print("\n=== MCTS vs MCTS ===")
    wA, wB, d = arena_mcts(
        netA, netB,
        args.games,
        args.simsA,
        args.simsB
    )

    print(f"\nNetA wins: {wA}")
    print(f"NetB wins: {wB}")
    print(f"Draws: {d}")

    if args.stockfish:

        print("\n=== vs Stockfish ===")

        w, l, d = arena_stockfish(
            netA,
            args.stockfish,
            args.depth,
            args.games,
            args.simsA
        )

        print(f"\nWins: {w}")
        print(f"Losses: {l}")
        print(f"Draws: {d}")

        elo = estimate_elo(w, l, d, args.stockfish_elo)
        print(f"\nEstimated Elo vs Stockfish: {elo:.1f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simsA", type=int, default=50)
    parser.add_argument("--simsB", type=int, default=50)

    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--stockfish_elo", type=int, default=2500)

    args = parser.parse_args()

    main(args)
