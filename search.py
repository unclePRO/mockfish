import time
import chess
from heuristics import evaluate_board

BIG = 100000            
moveCost = 10          
pieceValue = { #cost for each chess piece
    chess.PAWN: 100, 
    chess.KNIGHT: 320, 
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

goodMoves = []          # moves that worked well, possible moves.
moveCount = {}          # count of how often it was good (history heuristic)
stats = {"nodes": 0}    # for the telemetry bar

telemetry = {"depth": 0, "nodes": 0, "eval": 0.0, "time": 0.0}
def get_last_telemetry():
    return telemetry

def getScore(board, color):
    # evaluate_board is White-positive; flip it for Black
    s = evaluate_board(board)
    if color == chess.WHITE:
        return s
    return -s


def scoreMove(board, move):
    # captures first, then good quiet moves, then move count
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim == None:                     
            return 20000
        return 20000 + 10 * pieceValue[victim.piece_type] - pieceValue[attacker.piece_type]
    if move in goodMoves:
        return 9000
    return moveCount.get((move.from_square, move.to_square), 0)


def sortMoves(board):
    return sorted(board.legal_moves, key=lambda m: scoreMove(board, m), reverse=True)


def captureSearch(board, depth):
    # keep playing captures so we don't evaluate in the middle of a trade
    best = getScore(board, board.turn)
    if depth == 0:
        return best
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            best = max(best, -captureSearch(board, depth - 1))
            board.pop()
    return best


def getH(node):
    # h is lower when the side that just moved is better off
    if node.is_checkmate():
        return 0
    if node.is_stalemate() or node.is_insufficient_material():
        return BIG
    moverScore = -captureSearch(node, 2)
    if node.is_repetition(2):
        moverScore -= 50                        # anti-shuffling
    return BIG - moverScore


def minIndex(openList):
    minIndex = 0
    for i in range(len(openList)):
        if(openList[i][4] < openList[minIndex][4]):   # if f is less than min
            minIndex = i
    return minIndex


def reconstructPath(curr, closedList):
    path = []
    while(curr != None):
        for parent, child, move in closedList:
            if(child == curr):
                if move != None:
                    path.append(move)
                curr = parent
                break
    path.reverse()
    return path


def aStar(board, maxNodes, deadline):
    rootColor = board.turn
    rootKey = board.epd()                       # position as a string = key

    openList = []
    visitedList = []
    seen = {rootKey: 0}                         # transposition table: key -> best g
    expanded = 0

    # (parent, child, move, g, f, board) pair
    openList.append((None, rootKey, None, 0, 0, board.copy()))

    while(openList and expanded < maxNodes and time.time() < deadline):
        parent, child, move, g, f, node = openList.pop(minIndex(openList))
        visitedList.append((parent, child, move))
        expanded += 1

        # remember good quiet moves for move ordering
        if move != None and not board.is_capture(move) and parent == rootKey:
            if move not in goodMoves:
                goodMoves.append(move)
                if len(goodMoves) > 2:
                    goodMoves.pop(0)
            key = (move.from_square, move.to_square)
            moveCount[key] = moveCount.get(key, 0) + 1

        # goal: we checkmated the opponent
        if node.is_checkmate() and node.turn != rootColor:
            stats["nodes"] = expanded
            return reconstructPath(child, visitedList)

        if node.is_game_over():
            continue

        for m in sortMoves(node):
            bonus = min(scoreMove(node, m), 5000) // 100   # small ordering bonus
            node.push(m)
            key = node.epd()
            newG = g + moveCost
            if key not in seen or seen[key] > newG:        # new or cheaper
                seen[key] = newG
                h = getH(node)
                openList.append((child, key, m, newG, newG + h - bonus, node.copy()))
            node.pop()

    stats["nodes"] = expanded

    # if nothing left, then take the most promising node left
    if not openList:
        return []
    parent, child, move, g, f, node = openList[minIndex(openList)]
    visitedList.append((parent, child, move))
    return reconstructPath(child, visitedList)


def get_best_move(board, max_depth):
    telemetry["depth"] = len(path)
    telemetry["nodes"] = stats["nodes"]
    telemetry["eval"] = getScore(board, chess.WHITE)

    # returns a move (None if the game is over).
    if board.is_game_over():
        return None
    goodMoves.clear()
    moveCount.clear()

    path = aStar(board, 500 * max_depth, time.time() + 2.0)
    if path:
        return path[0]
    return sortMoves(board)[0]                  # fallback