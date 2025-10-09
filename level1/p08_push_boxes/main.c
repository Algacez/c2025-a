#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#define MAX_WIDTH 30
#define MAX_HEIGHT 20

#define PATH 0
#define WALL 1
#define PLAYER 2
#define BOX 4
#define TARGET 5
#define BOX_ON_TARGET 6

typedef struct {
    int maze[MAX_HEIGHT][MAX_WIDTH];
    int width;
    int height;
    int player_x;
    int player_y;
    int steps;
    int level;
    int boxes_on_target;
    int total_targets;
} GameState;

GameState game;

void clear_screen() {
    system("clear");
}

int get_char() {
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
}

int load_level(int level) {
    char filename[50];
    sprintf(filename, "level%d.txt", level);

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    game.width = 0;
    game.height = 0;
    game.steps = 0;
    game.level = level;
    game.boxes_on_target = 0;
    game.total_targets = 0;

    char line[MAX_WIDTH * 2 + 2];
    int y = 0;

    while (fgets(line, sizeof(line), file) && y < MAX_HEIGHT) {
        int x = 0;
        for (int i = 0; line[i] != '\0' && line[i] != '\n' && x < MAX_WIDTH; i++) {
            switch (line[i]) {
                case '#':
                    game.maze[y][x] = WALL;
                    x++;
                    break;
                case ' ':
                    game.maze[y][x] = PATH;
                    x++;
                    break;
                case 'P':
                    game.maze[y][x] = PLAYER;
                    game.player_x = x;
                    game.player_y = y;
                    x++;
                    break;
                case 'B':
                    game.maze[y][x] = BOX;
                    x++;
                    break;
                case 'T':
                    game.maze[y][x] = TARGET;
                    game.total_targets++;
                    x++;
                    break;
                case 'X':
                    game.maze[y][x] = BOX_ON_TARGET;
                    game.total_targets++;
                    game.boxes_on_target++;
                    x++;
                    break;
            }
        }
        if (x > game.width) {
            game.width = x;
        }
        y++;
    }

    game.height = y;
    fclose(file);
    return 1;
}

void display_maze() {
    clear_screen();
    printf("推箱子 - 第 %d 关\n", game.level);

    for (int y = 0; y < game.height; y++) {
        printf("  ");
        for (int x = 0; x < game.width; x++) {
            if (x == game.player_x && y == game.player_y) {
                if (game.maze[y][x] == TARGET) {
                    printf(" @");
                } else {
                    printf(" P");
                }
            } else {
                switch (game.maze[y][x]) {
                    case WALL:
                        printf("##");
                        break;
                    case PATH:
                        printf("  ");
                        break;
                    case BOX:
                        printf(" B");
                        break;
                    case TARGET:
                        printf(" T");
                        break;
                    case BOX_ON_TARGET:
                        printf(" ✅");
                        break;
                    default:
                        printf("  ");
                        break;
                }
            }
        }
        printf("\n");
    }
;
    printf("步数: %d\n", game.steps);
    printf("归位箱子: %d/%d\n", game.boxes_on_target, game.total_targets);
    printf("\n操作说明:\n");
    printf("\nwasd/箭头移动\n");
    printf("按r重置\n");
    printf("按q退出\n");
}

int is_valid(int x, int y) {
    return x >= 0 && x < game.width && y >= 0 && y < game.height;
}

int check_win() {
    return game.boxes_on_target == game.total_targets;
}

void save_score() {
    FILE *file = fopen("scores.txt", "a");
    if (file != NULL) {
        fprintf(file, "关卡 %d: %d 步\n", game.level, game.steps);
        fclose(file);
    }
}

void move_player(int dx, int dy) {
    int next_x = game.player_x + dx;
    int next_y = game.player_y + dy;

    if (!is_valid(next_x, next_y)) {
        return;
    }

    int next_cell = game.maze[next_y][next_x];

    if (next_cell == WALL) {
        return;
    }

    if (next_cell == BOX || next_cell == BOX_ON_TARGET) {
        int box_next_x = next_x + dx;
        int box_next_y = next_y + dy;

        if (!is_valid(box_next_x, box_next_y)) {
            return;
        }

        int box_next_cell = game.maze[box_next_y][box_next_x];

        if (box_next_cell != PATH && box_next_cell != TARGET) {
            return;
        }

        if (next_cell == BOX_ON_TARGET) {
            game.maze[next_y][next_x] = TARGET;
            game.boxes_on_target--;
        } else {
            game.maze[next_y][next_x] = PATH;
        }

        if (box_next_cell == TARGET) {
            game.maze[box_next_y][box_next_x] = BOX_ON_TARGET;
            game.boxes_on_target++;
        } else {
            game.maze[box_next_y][box_next_x] = BOX;
        }
    }

    game.player_x = next_x;
    game.player_y = next_y;
    game.steps++;
}

void handle_input() {
    int key = get_char();

    if (key == 27) {
        int next1 = get_char();
        if (next1 == 91) {
            int next2 = get_char();
            switch (next2) {
                case 'A': key = 'w'; break;
                case 'B': key = 's'; break;
                case 'C': key = 'd'; break;
                case 'D': key = 'a'; break;
                default: return;
            }
        } else {
            return;
        }
    }

    switch (key) {
        case 'w': case 'W':
            move_player(0, -1);
            break;
        case 's': case 'S':
            move_player(0, 1);
            break;
        case 'a': case 'A':
            move_player(-1, 0);
            break;
        case 'd': case 'D':
            move_player(1, 0);
            break;
        case 'r': case 'R':
            load_level(game.level);
            break;
        case 'q': case 'Q':
            printf("\n游戏退出！\n");
            exit(0);
    }
}

int main() {
    int current_level = 1;

    printf("\n按任意键开始游戏...\n");
    get_char();

    while (1) {
        if (!load_level(current_level)) {
            if (current_level == 1) {
                printf("\n错误：找不到关卡文件\n");
                return 1;
            }
            else {
                printf("恭喜！你已完成所有关卡！\n");
                break;
            }
        }

        while (1) {
            display_maze();

            if (check_win()) {
                save_score();
                printf("\n🎉 恭喜过关！用了 %d 步！\n", game.steps);
                printf("按任意键进入下一关...\n");
                get_char();
                current_level++;
                break;
            }

            handle_input();
        }
    }

    return 0;
}