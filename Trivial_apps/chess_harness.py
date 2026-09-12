
import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Piece weights ----------
PIECE_WEIGHTS = {
    'P': 0.1,   # pawn
    'N': 0.3,   # knight
    'B': 0.4,   # bishop
    'R': 0.5,   # rook
    'Q': 1.0,   # queen
    'K': 0.9,   # king
}

def get_weight(symbol):
    """Return weight for a piece symbol (case-insensitive)."""
    return PIECE_WEIGHTS[symbol.upper()]

# ---------- TSP routing (bucket sort by phase) ----------
def tsp_route_complex(points):
    """
    points: list of (x,y) coordinates (0‑based)
    Returns order of indices sorted by angle from origin.
    """
    if len(points) <= 1:
        return list(range(len(points)))
    angles = np.angle(np.array([complex(x, y) for x, y in points]))
    buckets = [[] for _ in range(360)]
    for idx, a in enumerate(angles):
        a_norm = a + PI if a < 0 else a
        b = int((a_norm / (2 * PI)) * 360) % 360
        buckets[b].append(idx)
    order = []
    for b in buckets:
        order.extend(b)
    return order

# ---------- Merge sort on weights ----------
def merge_sort_by_weight(pieces):
    """
    pieces: list of (x, y, symbol)
    Returns list sorted by weight ascending (merge sort, O(K log K)).
    """
    if len(pieces) <= 1:
        return pieces
    mid = len(pieces) // 2
    left = merge_sort_by_weight(pieces[:mid])
    right = merge_sort_by_weight(pieces[mid:])
    # merge
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if get_weight(left[i][2]) <= get_weight(right[j][2]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# ---------- Board evaluation ----------
def evaluate_board(board):
    """
    board: 8x8 list of lists, each element is a piece symbol or None.
    Returns a scalar score.
    Score = weighted_sum * (1 + 0.5 * entropy_factor)
    where entropy_factor = 1 / (1 + TSP_tour_length)
    """
    pieces = []
    for i in range(8):
        for j in range(8):
            if board[i][j] is not None:
                pieces.append((i, j, board[i][j]))
    if not pieces:
        return 0.0

    # Weighted sum
    weight_sum = sum(get_weight(p[2]) for p in pieces)

    # TSP entropy: sort by angle and compute total distance
    pts = [(p[1], p[0]) for p in pieces]   # (x, y)
    order = tsp_route_complex(pts)
    total_dist = 0.0
    for idx in range(len(order)):
        i1 = order[idx]
        i2 = order[(idx + 1) % len(order)]
        x1, y1 = pts[i1]
        x2, y2 = pts[i2]
        total_dist += math.hypot(x2 - x1, y2 - y1)
    entropy_factor = 1.0 / (1.0 + total_dist)

    # Combine
    score = weight_sum * (1.0 + 0.5 * entropy_factor)
    return score

# ---------- Find best diagonal move ----------
def find_best_diagonal_move(board):
    """
    For each piece, try all diagonal squares.
    Returns (best_board, best_score).
    """
    best_score = evaluate_board(board)
    best_board = [row[:] for row in board]

    for i in range(8):
        for j in range(8):
            if board[i][j] is not None:
                piece = board[i][j]
                for di, dj in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 8 and 0 <= nj < 8:
                        # Try move
                        new_board = [row[:] for row in board]
                        new_board[ni][nj] = piece
                        new_board[i][j] = None
                        score = evaluate_board(new_board)
                        if score > best_score:
                            best_score = score
                            best_board = new_board
    return best_board, best_score

# ---------- Main ----------
def main():
    # Initialise standard chess starting position (white pieces)
    board = [[None] * 8 for _ in range(8)]
    back_rank = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
    for col in range(8):
        board[0][col] = back_rank[col]   # white back rank
        board[1][col] = 'P'              # white pawns
        board[6][col] = 'p'              # black pawns (for demo we keep them as lowercase)
        board[7][col] = back_rank[col].lower()  # black back rank (lowercase)
    # We'll use uppercase for lookup, so lowercases are fine.

    print("Initial board:")
    for row in board:
        print(' '.join([c if c is not None else '.' for c in row]))

    print("\nInitial score:", evaluate_board(board))

    best_board, best_score = find_best_diagonal_move(board)
    print("Best score after diagonal move:", best_score)
    print("Board after best diagonal move:")
    for row in best_board:
        print(' '.join([c if c is not None else '.' for c in row]))

if __name__ == "__main__":
    main()
