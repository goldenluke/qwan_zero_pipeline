import chess
import random

def best_move(fen):

    board = chess.Board(fen)

    moves = list(board.legal_moves)

    if not moves:
        return None

    return random.choice(moves).uci()
