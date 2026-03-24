import streamlit as st
import chess
import streamlit.components.v1 as components

from backend.engine import best_move

st.title("QWAN Chess Engine")

if "fen" not in st.session_state:
    st.session_state.fen = chess.Board().fen()

board = chess.Board(st.session_state.fen)

html_board = f"""
<!DOCTYPE html>
<html>
<head>

<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/chessboard.js/1.0.0/chessboard.min.css"/>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.13.4/chess.min.js"></script>

<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard.js/1.0.0/chessboard.min.js"></script>

</head>

<body>

<div id="board" style="width:500px"></div>

<script>

var board = null
var game = new Chess("{board.fen()}")

function onDragStart (source, piece, position, orientation) {{

  if (game.game_over()) return false

  if (piece.search(/^b/) !== -1) return false

}}

function onDrop(source, target) {{

  var move = game.move({{
    from: source,
    to: target,
    promotion: 'q'
  }})

  if (move === null) return 'snapback'

  window.parent.postMessage({{
      type: "move",
      fen: game.fen()
  }}, "*")

}}

var config = {{
  draggable: true,
  position: game.fen(),
  onDragStart: onDragStart,
  onDrop: onDrop
}}

board = Chessboard('board', config)

</script>

</body>
</html>
"""

move_data = components.html(html_board, height=550)

st.write("Current FEN")
st.code(st.session_state.fen)

if st.button("Reset Game"):

    st.session_state.fen = chess.Board().fen()
