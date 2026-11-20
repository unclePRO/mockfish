import time
import math
import chess

from heuristics import evaluate_board, CHECKMATE_SCORE
from search import sortMoves as order_moves

# Algorithm 1: Pure Minimax (Unpruned Exhaustive Game Tree Search)
minimax_nodes = 0

def minimax_raw(board: chess.Board, depth: int, is_max: bool) -> float:
    """Pure unpruned Minimax search exploring the entire game tree to depth."""
    global minimax_nodes
    minimax_nodes += 1

    if board.is_checkmate():
        return (-CHECKMATE_SCORE - depth) if board.turn == chess.WHITE else (CHECKMATE_SCORE + depth)
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    if depth == 0:
        return evaluate_board(board)

    moves = list(board.legal_moves)
    if is_max:
        max_eval = -math.inf
        for m in moves:
            board.push(m)
            val = minimax_raw(board, depth - 1, False)
            board.pop()
            max_eval = max(max_eval, val)
        return max_eval
    else:
        min_eval = math.inf
        for m in moves:
            board.push(m)
            val = minimax_raw(board, depth - 1, True)
            board.pop()
            min_eval = min(min_eval, val)
        return min_eval


# Algorithm 2: Alpha-Beta Pruning (Unordered / Arbitrary Move Order)
ab_unordered_nodes = 0

def alpha_beta_unordered(board: chess.Board, depth: int, alpha: float, beta: float, is_max: bool) -> float:
    """Alpha-Beta pruning without heuristic move ordering."""
    global ab_unordered_nodes
    ab_unordered_nodes += 1

    if board.is_checkmate():
        return (-CHECKMATE_SCORE - depth) if board.turn == chess.WHITE else (CHECKMATE_SCORE + depth)
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw() or board.is_repetition(2):
        return 0.0
    if depth == 0:
        return evaluate_board(board)

    moves = list(board.legal_moves)
    if is_max:
        max_eval = -math.inf
        for m in moves:
            board.push(m)
            val = alpha_beta_unordered(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for m in moves:
            board.push(m)
            val = alpha_beta_unordered(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval


# Algorithm 3: Alpha-Beta Pruning + Heuristic Move Ordering (Mockfish Core)
ab_ordered_nodes = 0

def alpha_beta_ordered(board: chess.Board, depth: int, alpha: float, beta: float, is_max: bool) -> float:
    """Alpha-Beta pruning with heuristic move ordering (MVV-LVA, promotions, checks, center)."""
    global ab_ordered_nodes
    ab_ordered_nodes += 1

    if board.is_checkmate():
        return (-CHECKMATE_SCORE - depth) if board.turn == chess.WHITE else (CHECKMATE_SCORE + depth)
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw() or board.is_repetition(2):
        return 0.0
    if depth == 0:
        return evaluate_board(board)

    moves = order_moves(board)
    if is_max:
        max_eval = -math.inf
        for m in moves:
            board.push(m)
            val = alpha_beta_ordered(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = math.inf
        for m in moves:
            board.push(m)
            val = alpha_beta_ordered(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval


# Algorithm 4: Iterative Deepening Depth-First Search (IDDFS)
iddfs_nodes = 0

def iterative_deepening(board: chess.Board, max_depth: int) -> tuple[chess.Move, float, int]:
    """Iterative Deepening Search iteratively running Alpha-Beta from depth 1 to max_depth."""
    global iddfs_nodes, ab_ordered_nodes
    iddfs_nodes = 0
    best_move = list(board.legal_moves)[0]
    best_eval = 0.0
    is_max = (board.turn == chess.WHITE)

    for d in range(1, max_depth + 1):
        ab_ordered_nodes = 0
        ordered = order_moves(board)
        curr_best_eval = -math.inf if is_max else math.inf
        curr_best_move = ordered[0]
        alpha = -math.inf
        beta = math.inf

        for m in ordered:
            board.push(m)
            score = alpha_beta_ordered(board, d - 1, alpha, beta, not is_max)
            board.pop()

            if is_max:
                if score > curr_best_eval:
                    curr_best_eval = score
                    curr_best_move = m
                alpha = max(alpha, score)
            else:
                if score < curr_best_eval:
                    curr_best_eval = score
                    curr_best_move = m
                beta = min(beta, score)

        iddfs_nodes += ab_ordered_nodes
        best_move = curr_best_move
        best_eval = curr_best_eval

    return best_move, best_eval, iddfs_nodes


# Benchmark Runner
def run_benchmarks():
    global minimax_nodes, ab_unordered_nodes, ab_ordered_nodes

    test_positions = [
        ("Middlegame (Italian Game)", chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 4")),
        ("Tactical Exchange Position", chess.Board("r1b1k2r/pppp1ppp/8/4N3/1b1nn3/2N5/PPPP1PPP/R1BQKB1R w KQkq - 0 7")),
    ]

    print(" MOCKFISH ALGORITHM PERFORMANCE COMPARISON ")

    for pos_name, board in test_positions:
        print(f"\n--- Testing Position: {pos_name} ---")
        print(f"    FEN: {board.fen()}")
        print(f"    Legal Moves Available: {len(list(board.legal_moves))}")
        print("-----------------------------------------------------------------------------------------")
        print(f"{'Algorithm':<36} | {'Depth':<5} | {'Nodes':<10} | {'Time (ms)':<10} | {'Eff. Branching'}")
        print("-----------------------------------------------------------------------------------------")

        for depth in [1, 2, 3]:
            # 1. Pure Minimax
            minimax_nodes = 0
            t0 = time.perf_counter()
            minimax_raw(board, depth, board.turn == chess.WHITE)
            t_minimax = (time.perf_counter() - t0) * 1000
            b_minimax = minimax_nodes ** (1.0 / depth) if depth > 0 else 1.0
            print(f"{'Pure Minimax (Exhaustive)':<36} | {depth:<5} | {minimax_nodes:<10} | {t_minimax:<10.2f} | {b_minimax:.2f}")

            # 2. Alpha-Beta (Arbitrary / Unordered)
            ab_unordered_nodes = 0
            t0 = time.perf_counter()
            alpha_beta_unordered(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
            t_ab_unord = (time.perf_counter() - t0) * 1000
            b_ab_unord = ab_unordered_nodes ** (1.0 / depth) if depth > 0 else 1.0
            print(f"{'Alpha-Beta (Unordered)':<36} | {depth:<5} | {ab_unordered_nodes:<10} | {t_ab_unord:<10.2f} | {b_ab_unord:.2f}")

            # 3. Alpha-Beta + Heuristic Move Ordering (Mockfish)
            ab_ordered_nodes = 0
            t0 = time.perf_counter()
            alpha_beta_ordered(board, depth, -math.inf, math.inf, board.turn == chess.WHITE)
            t_ab_ord = (time.perf_counter() - t0) * 1000
            b_ab_ord = ab_ordered_nodes ** (1.0 / depth) if depth > 0 else 1.0
            print(f"{'Alpha-Beta + Move Ordering':<36} | {depth:<5} | {ab_ordered_nodes:<10} | {t_ab_ord:<10.2f} | {b_ab_ord:.2f}")

            # 4. Iterative Deepening (IDDFS)
            t0 = time.perf_counter()
            _, _, iddfs_total_nodes = iterative_deepening(board, depth)
            t_iddfs = (time.perf_counter() - t0) * 1000
            b_iddfs = iddfs_total_nodes ** (1.0 / depth) if depth > 0 else 1.0
            print(f"{'Iterative Deepening DFS (IDDFS)':<36} | {depth:<5} | {iddfs_total_nodes:<10} | {t_iddfs:<10.2f} | {b_iddfs:.2f}")

if __name__ == "__main__":
    run_benchmarks()
