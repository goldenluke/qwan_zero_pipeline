import math
import random
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# UTILIDADES
# ============================================================

def board_to_tensor(board):
    tensor = np.zeros((12,8,8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        r = sq // 8
        c = sq % 8
        idx = piece.piece_type - 1
        if not piece.color:
            idx += 6
        tensor[idx,r,c] = 1
    return torch.tensor(tensor)

def move_to_index(move):
    return hash(move) % 4672

# ============================================================
# REDE
# ============================================================

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(12,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU()
        )

        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8,512),
            nn.ReLU(),
            nn.Linear(512,4672)
        )

        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.conv(x)
        return self.policy(x), self.value(x)

# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self,size=200000):
        self.buffer = deque(maxlen=size)

    def add(self,item):
        self.buffer.append(item)

    def sample(self,batch):
        idx = np.random.choice(len(self.buffer),batch)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)

# ============================================================
# MCTS COM MODULAÇÃO CRÍTICA
# ============================================================

class MCTS:
    def __init__(self, net, sims=50, c_puct=1.5, dirichlet_alpha=0.3, temperature=1.0):
        self.net = net
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = temperature

        self.Q = {}
        self.N = {}
        self.P = {}

    def search(self, board):
        for _ in range(self.sims):
            self._simulate(board.copy())

        state = board.fen()
        legal = list(board.legal_moves)

        counts = np.array([self.N.get((state,a),0) for a in legal], dtype=np.float32)

        if self.temperature != 1.0:
            counts = counts ** (1/self.temperature)

        probs = counts / (np.sum(counts)+1e-8)
        return dict(zip(legal, probs))

    def _simulate(self, board):
        state = board.fen()

        if board.is_game_over():
            return self._result(board)

        if state not in self.P:
            return self._expand(board)

        legal = list(board.legal_moves)
        total_n = sum(self.N.get((state,a),0) for a in legal)

        best = None
        best_score = -1e9

        for a in legal:
            q = self.Q.get((state,a),0)
            n = self.N.get((state,a),0)
            p = self.P[state][a]

            u = self.c_puct * p * np.sqrt(total_n+1)/(1+n)
            s = q + u

            if s > best_score:
                best_score = s
                best = a

        board.push(best)
        v = self._simulate(board)

        self.N[(state,best)] = self.N.get((state,best),0)+1
        self.Q[(state,best)] = (
            self.Q.get((state,best),0)
            + (v - self.Q.get((state,best),0))
            / self.N[(state,best)]
        )

        return -v

    def _expand(self, board):
        state = board.fen()

        x = board_to_tensor(board).unsqueeze(0).to(DEVICE)
        policy_logits, value = self.net(x)

        policy = torch.softmax(policy_logits,1).detach().cpu().numpy()[0]
        legal = list(board.legal_moves)

        probs = np.array([policy[move_to_index(a)] for a in legal])

        noise = np.random.dirichlet([self.dirichlet_alpha]*len(probs))
        probs = 0.75*probs + 0.25*noise
        probs /= np.sum(probs)+1e-8

        self.P[state] = dict(zip(legal, probs))
        return value.item()

    def _result(self,board):
        r = board.result()
        if r=="1-0": return 1
        if r=="0-1": return -1
        return 0

# ============================================================
# SELF PLAY
# ============================================================

def self_play(net, buffer, games, sims, c_puct=1.5, dirichlet_alpha=0.3, temperature=1.0):

    for _ in tqdm(range(games), desc="Self-play"):

        board = chess.Board()

        mcts = MCTS(
            net,
            sims=sims,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            temperature=temperature
        )

        while not board.is_game_over():

            pi = mcts.search(board)
            moves = list(pi.keys())
            probs = np.array(list(pi.values()))

            move = np.random.choice(moves,p=probs)

            policy_vector = np.zeros(4672, dtype=np.float32)
            for m, p in zip(moves, probs):
                policy_vector[move_to_index(m)] = p

            buffer.add((board_to_tensor(board), policy_vector, 0.0))
            board.push(move)

# ============================================================
# TRAINING
# ============================================================

def train(net, buffer, batches, batch_size):

    opt = optim.Adam(net.parameters(), lr=1e-3)

    for _ in range(batches):

        if len(buffer) < batch_size:
            return

        batch = buffer.sample(batch_size)
        states, policies, values = zip(*batch)

        states = torch.stack(states).to(DEVICE)
        policies = torch.from_numpy(np.array(policies)).float().to(DEVICE)
        values = torch.tensor(values).float().to(DEVICE)

        p_logits, v_pred = net(states)

        loss_p = -(policies * torch.log_softmax(p_logits,1)).sum(1).mean()
        loss_v = ((v_pred.squeeze()-values)**2).mean()

        loss = loss_p + loss_v

        opt.zero_grad()
        loss.backward()
        opt.step()
