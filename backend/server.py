from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.engine import best_move

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/move")
def move(fen: str):

    move = best_move(fen)

    return {"move": move}
