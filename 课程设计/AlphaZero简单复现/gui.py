# -*- coding: utf-8 -*-
# gui.py
import sys
import pygame
import torch
from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

pygame.init()

BOARD_SIZE = 9
N_IN_ROW = 5
CELL_SIZE = 40
LINE_WIDTH = 1
SCREEN_WIDTH = (BOARD_SIZE + 1) * CELL_SIZE
SCREEN_HEIGHT = (BOARD_SIZE + 1) * CELL_SIZE
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

try:
    bg_original = pygame.image.load('assets/background.jpg')
    background = pygame.transform.scale(bg_original, (SCREEN_WIDTH, SCREEN_HEIGHT))
except:
    background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    background.fill((200, 100, 100))

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("AlphaZero 五子棋 (9x9)")

board_game = Board(width=BOARD_SIZE, height=BOARD_SIZE, n_in_row=N_IN_ROW)
board_game.init_board()

model_file = 'best_policy.model' 
use_gpu = torch.cuda.is_available()
print(f"GUI AI 模式, 使用设备: {'GPU' if use_gpu else 'CPU'}")

try:
    policy_value_net = PolicyValueNet(BOARD_SIZE, BOARD_SIZE, model_file=model_file, use_gpu=use_gpu)
    print("AI 模型加载成功！")
except Exception as e:
    print(f"未找到模型文件或加载失败: {e}，AI 将使用随机初始化权重的策略。")
    policy_value_net = PolicyValueNet(BOARD_SIZE, BOARD_SIZE, model_file=None, use_gpu=use_gpu)

mcts_player = MCTSPlayer(policy_value_net.policy_value_fn, c_puct=5, n_playout=400)

HUMAN_PLAYER = 1
AI_PLAYER = 2
board_game.players = [HUMAN_PLAYER, AI_PLAYER] 

GAME_OVER = False
WINNER = None

def draw_board():
    for i in range(BOARD_SIZE):
        start = (CELL_SIZE, (i + 1) * CELL_SIZE)
        end = (SCREEN_WIDTH - CELL_SIZE, (i + 1) * CELL_SIZE)
        pygame.draw.line(screen, BLACK_COLOR, start, end, LINE_WIDTH)
        start = ((i + 1) * CELL_SIZE, CELL_SIZE)
        end = ((i + 1) * CELL_SIZE, SCREEN_HEIGHT - CELL_SIZE)
        pygame.draw.line(screen, BLACK_COLOR, start, end, LINE_WIDTH)

def draw_stones():
    radius = CELL_SIZE // 2 - 4
    for move, player in board_game.states.items():
        row = move // BOARD_SIZE
        col = move % BOARD_SIZE
        
        stone_x = (col + 1) * CELL_SIZE
        stone_y = (row + 1) * CELL_SIZE

        if player == HUMAN_PLAYER: 
            color = BLACK_COLOR
        else: 
            color = WHITE_COLOR

        pygame.draw.circle(screen, color, (stone_x, stone_y), radius)
        
        if move == board_game.last_move:
            pygame.draw.rect(screen, (255, 0, 0), (stone_x-5, stone_y-5, 10, 10))

def get_grid_pos(mouse_x, mouse_y):
    col = round((mouse_x - CELL_SIZE) / CELL_SIZE)
    row = round((mouse_y - CELL_SIZE) / CELL_SIZE)
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None

def main():
    global GAME_OVER, WINNER
    running = True
    clock = pygame.time.Clock()

    board_game.init_board()
    
    while running:
        current_player = board_game.get_current_player()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not GAME_OVER and current_player == HUMAN_PLAYER:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    grid = get_grid_pos(*mouse_pos)
                    
                    if grid is not None:
                        row, col = grid
                        move = board_game.location_to_move((row, col))
                        
                        if move in board_game.availables:
                            board_game.do_move(move)
                            end, winner = board_game.game_end()
                            if end:
                                GAME_OVER = True
                                WINNER = winner
        
        if not GAME_OVER and current_player == AI_PLAYER:
            screen.blit(background, (0, 0))
            draw_board()
            draw_stones()
            pygame.display.flip()
            
            ai_move = mcts_player.get_action(board_game)
            board_game.do_move(ai_move)
            
            end, winner = board_game.game_end()
            if end:
                GAME_OVER = True
                WINNER = winner

        screen.blit(background, (0, 0))
        draw_board()
        draw_stones()

        font = pygame.font.SysFont('Arial', 24)

        if not GAME_OVER:
            if current_player == HUMAN_PLAYER:
                turn_text = "Turn: Black (You)"
            else:
                turn_text = "Turn: White (AI)"
            txt_surf = font.render(turn_text, True, (0, 0, 0))
            screen.blit(txt_surf, (10, 8))
        else:
            if WINNER == HUMAN_PLAYER:
                txt = "Winner: Black (You)!"
            elif WINNER == AI_PLAYER:
                txt = "Winner: White (AI)!"
            else:
                txt = "Draw (Tie)!"
            
            txt_winner = font.render(txt, True, (255, 0, 0))
            screen.blit(txt_winner, (10, 8))
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                board_game.init_board()
                mcts_player.reset_player()
                GAME_OVER = False
                WINNER = None
            
            reset_txt = font.render("Press 'R' to Restart", True, (0, 0, 255))
            screen.blit(reset_txt, (SCREEN_WIDTH - 200, 8))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()