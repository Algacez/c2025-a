# ai.py
import sys

BOARD_SIZE = 15
EMPTY = 0
BLACK_STONE = 1
WHITE_STONE = 2

sys.setrecursionlimit(10000)

def evaluate(board, player):

    directions = [(1,0),(0,1),(1,1),(1,-1)]
    opponent = BLACK_STONE if player == WHITE_STONE else WHITE_STONE

    W_FIVE = 100000
    W_FOUR_OPEN = 10000
    W_FOUR_HALF = 2500
    W_THREE_OPEN = 1500
    W_THREE_HALF = 200
    W_TWO_OPEN = 100
    W_GAPPED_BONUS = 400
    OPPONENT_WEIGHT = 1.2
    IMMEDIATE_THREAT_PENALTY = 1000000

    def in_bounds(x,y):
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def analyze_from(i,j,dx,dy,cur):
        opp = BLACK_STONE if cur == WHITE_STONE else WHITE_STONE

        cnt = 0
        x = i
        y = j
        while in_bounds(x,y) and board[x][y] == cur:
            cnt += 1
            x += dx
            y += dy
        right_x, right_y = x, y

        left_x, left_y = i - dx, j - dy

        left_empty = (in_bounds(left_x,left_y) and board[left_x][left_y] == EMPTY)
        left_blocked = (not in_bounds(left_x,left_y)) or (in_bounds(left_x,left_y) and board[left_x][left_y] == opp)

        right_empty = (in_bounds(right_x,right_y) and board[right_x][right_y] == EMPTY)
        right_blocked = (not in_bounds(right_x,right_y)) or (in_bounds(right_x,right_y) and board[right_x][right_y] == opp)

        open_ends = (1 if left_empty else 0) + (1 if right_empty else 0)

        score = 0
        threats = set()
        immediate_win = False

        if cnt >= 5:
            score += W_FIVE
            immediate_win = True
        elif cnt == 4:
            if open_ends == 2:
                score += W_FOUR_OPEN
                if cur == opponent:
                    immediate_win = True
                if left_empty:
                    threats.add((left_x,left_y))
                if right_empty:
                    threats.add((right_x,right_y))
            elif open_ends == 1:
                score += W_FOUR_HALF
                if left_empty:
                    threats.add((left_x,left_y))
                if right_empty:
                    threats.add((right_x,right_y))
                if cur == opponent:
                    immediate_win = True
            else:
                score += W_FOUR_HALF // 4
        elif cnt == 3:
            if open_ends == 2:
                score += W_THREE_OPEN
                if left_empty:
                    threats.add((left_x,left_y))
                if right_empty:
                    threats.add((right_x,right_y))
            elif open_ends == 1:
                score += W_THREE_HALF
                if left_empty:
                    threats.add((left_x,left_y))
                if right_empty:
                    threats.add((right_x,right_y))
            else:
                score += 30
        elif cnt == 2:
            if open_ends == 2:
                score += W_TWO_OPEN
            elif open_ends == 1:
                score += 8
            else:
                score += 2
        elif cnt == 1:
            if open_ends == 2:
                score += 3

        if in_bounds(right_x,right_y) and board[right_x][right_y] == EMPTY:
            after_gap_x = right_x + dx
            after_gap_y = right_y + dy
            if in_bounds(after_gap_x, after_gap_y) and board[after_gap_x][after_gap_y] == cur:
                score += W_GAPPED_BONUS * (cnt)
                threats.add((right_x,right_y))
                if cur == opponent and (cnt + 1) >= 4:
                    immediate_win = True

        if in_bounds(left_x,left_y) and board[left_x][left_y] == EMPTY:
            before_gap_x = left_x - dx
            before_gap_y = left_y - dy
            if in_bounds(before_gap_x, before_gap_y) and board[before_gap_x][before_gap_y] == cur:
                score += W_GAPPED_BONUS * (cnt)
                threats.add((left_x,left_y))
                if cur == opponent and (cnt + 1) >= 4:
                    immediate_win = True

        return score, threats, immediate_win

    def get_aggregate(cur):
        total = 0
        threat_positions = set()
        has_immediate = False
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i][j] != cur:
                    continue
                for dx,dy in directions:
                    prev_x, prev_y = i - dx, j - dy
                    if in_bounds(prev_x, prev_y) and board[prev_x][prev_y] == cur:
                        continue
                    add, threats, immediate = analyze_from(i,j,dx,dy,cur)
                    total += add
                    threat_positions |= threats
                    if immediate:
                        has_immediate = True
        return total, threat_positions, has_immediate

    my_score, my_threats, my_immediate = get_aggregate(player)
    opp_score, opp_threats, opp_immediate = get_aggregate(opponent)

    if opp_immediate:
        return -IMMEDIATE_THREAT_PENALTY

    threat_penalty = len(opp_threats) * 2000
    score = (my_score - OPPONENT_WEIGHT * opp_score) - threat_penalty

    return int(score)

def get_possible_moves(board, radius = 2):
    moves = set()
    has_piece = False
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != EMPTY:
                has_piece = True
                r0 = max(0, i - radius)
                r1 = min(BOARD_SIZE - 1, i + radius)
                c0 = max(0, j - radius)
                c1 = min(BOARD_SIZE - 1, j + radius)
                for x in range(r0, r1 + 1):
                    for y in range(c0, c1 + 1):
                        if board[x][y] == EMPTY:
                            moves.add((x, y))
    if not has_piece:
        center = BOARD_SIZE // 2
        return [(center, center)]
    return list(moves)

def minimax(board, depth, alpha, beta, maximizing_player, current_player, original_player):
    if depth == 0:
        return evaluate(board, original_player)

    possible_moves = get_possible_moves(board)

    if not possible_moves:
        return evaluate(board, original_player)
    
    opponent = BLACK_STONE if current_player == WHITE_STONE else WHITE_STONE

    if maximizing_player:
        max_eval = -float('inf')

        for move in possible_moves:
            i, j = move
            board[i][j] = current_player
            eval_score = minimax(board, depth - 1, alpha, beta, False, opponent, original_player)
            board[i][j] = EMPTY  # 回溯
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    
    else:
        min_eval = float('inf')
        for move in possible_moves:
            i, j = move
            board[i][j] = current_player
            eval_score = minimax(board, depth - 1, alpha, beta, True, opponent, original_player)
            board[i][j] = EMPTY  # 回溯
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, player, depth = 3):
    best_score = -float('inf')
    best_move = None
    possible_moves = get_possible_moves(board)
    for move in possible_moves:
        i, j = move
        board[i][j] = player
        opponent = BLACK_STONE if player == WHITE_STONE else WHITE_STONE
        score = minimax(board, depth - 1, -float('inf'), float('inf'), False, opponent, player)
        board[i][j] = EMPTY
        if score > best_score:
            best_score = score
            best_move = move
    return best_move