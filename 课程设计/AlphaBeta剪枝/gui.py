# gui.py
import sys
import pygame

import ai

pygame.init()

BOARD_SIZE = 15
CELL_SIZE = 40
LINE_WIDTH = 2

SCREEN_WIDTH = (BOARD_SIZE + 1) * CELL_SIZE
SCREEN_HEIGHT = (BOARD_SIZE + 1) * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

bg_original = pygame.image.load('assets/background.jpg')
background = pygame.transform.scale(bg_original, 
                                    (SCREEN_WIDTH, SCREEN_HEIGHT))

screen = pygame.display.set_mode((SCREEN_WIDTH, 
                                  SCREEN_HEIGHT))
pygame.display.set_caption("五子棋AI对战")

EMPTY = 0
BLACK_STONE = 1
WHITE_STONE = 2
board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

CURRENT_TURN = BLACK_STONE

def draw_board():
    for i in range(BOARD_SIZE):
        x = (i + 1) * CELL_SIZE
        y = x

        pygame.draw.line(screen, 
                         BLACK, 
                         (CELL_SIZE, x), 
                         (SCREEN_WIDTH - CELL_SIZE, y), 
                         LINE_WIDTH)
        pygame.draw.line(screen, 
                         BLACK, 
                         (y, CELL_SIZE), 
                         (x, SCREEN_HEIGHT - CELL_SIZE), 
                         LINE_WIDTH)

def draw_stones():
    radius = CELL_SIZE // 2 - 4
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            current_stone = board[row][col]
            if current_stone == 0:
                continue
            stone_x = (col + 1) * CELL_SIZE
            stone_y = (row + 1) * CELL_SIZE

            if current_stone == BLACK_STONE: color = BLACK
            else: color = WHITE

            pygame.draw.circle(screen, color, (stone_x, stone_y), radius)
                
def place_stone(row, col):
    global CURRENT_TURN

    if board[row][col] != EMPTY:
        return False
    
    board[row][col] = CURRENT_TURN
    if CURRENT_TURN == BLACK_STONE: 
        CURRENT_TURN = WHITE_STONE
    else: 
        CURRENT_TURN = BLACK_STONE

    return True

def get_grid_pos(mouse_x, mouse_y):

    col = round((mouse_x - CELL_SIZE) / CELL_SIZE)
    row = round((mouse_y - CELL_SIZE) / CELL_SIZE)

    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    
    return None

def check_win(last_row, last_col, player):
    directions = [
        (1,0),
        (0,1),
        (1,1),
        (1,-1)
    ]

    for d_x, d_y in directions:
        count = 1
        for i in range(1,5):
            current_row = last_row + i * d_x
            current_col = last_col + i * d_y
            if 0 <= current_row < BOARD_SIZE and 0 <= current_col < BOARD_SIZE and board[current_row][current_col] == player:
                count += 1
            else:
                break

        for i in range(1,5):
            current_row = last_row - i * d_x
            current_col = last_col - i * d_y
            if 0 <= current_row < BOARD_SIZE and 0 <= current_col < BOARD_SIZE and board[current_row][current_col] == player:
                count += 1
            else:
                break

        if count >= 5:
            return True

    return False 

GAME_OVER = False
WINNER = None

def main():
    global CURRENT_TURN, GAME_OVER, WINNER
    running = True
    clock = pygame.time.Clock()
    while running:
        for status in pygame.event.get():
            if status.type == pygame.QUIT:
                running = False
            
            if GAME_OVER:
                continue

            if status.type == pygame.MOUSEBUTTONDOWN and status.button == 1 and CURRENT_TURN == BLACK_STONE:
                mouse_pos = pygame.mouse.get_pos()
                grid = get_grid_pos(*mouse_pos)
                if grid is None:
                    continue

                row, col = grid
                last_player = CURRENT_TURN

                if place_stone(row, col):
                    if check_win(row, col, last_player):
                        GAME_OVER = True
                        WINNER = last_player

        if CURRENT_TURN == WHITE_STONE and not GAME_OVER:
            ai_move = ai.get_best_move(board, WHITE_STONE)
            if ai_move:
                row, col = ai_move
                last_player = CURRENT_TURN
                place_stone(row, col)
                if check_win(row, col, last_player):
                    GAME_OVER = True
                    WINNER = last_player


        screen.blit(background, (0, 0))
        draw_board()
        draw_stones()

        font = pygame.font.SysFont(None, 24)

        if not GAME_OVER:
            if CURRENT_TURN == BLACK_STONE:
                turn_text = "Black"
            else:
                turn_text = "White"

            txt_surf = font.render(f"Current: {turn_text}", True, (0, 0, 0))
            screen.blit(txt_surf, (10, 8))

        else:
            if WINNER == 1:
                winner_color = "Black" 

            else:
                winner_color = "White"

            txt_winner = font.render(f"GAME OVER! Winner: {winner_color}", True, (0, 0, 0))
            screen.blit(txt_winner, (10, 8))
            
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()