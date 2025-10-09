#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <termios.h>
#include <unistd.h>

#define WIDTH 31
#define HEIGHT 15

#define PATH 0
#define WALL 1
#define PLAYER 2
#define EXIT 3

#define START_X 1
#define START_Y 1
#define EXIT_X (WIDTH - 2)
#define EXIT_Y (HEIGHT - 2)

int maze[HEIGHT][WIDTH];
int player_x, player_y;

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

void initialize_maze();
void generate_maze_recursive(int x, int y);
void display_maze();
void handle_input();
int is_valid(int x, int y);

// 自顶向下
int main() {
    srand((unsigned int)time(NULL));

    initialize_maze();
    generate_maze_recursive(START_X, START_Y);

    player_x = START_X;
    player_y = START_Y;
    maze[EXIT_Y][EXIT_X] = EXIT;

    while (1) {
        maze[player_y][player_x] = PLAYER;
        display_maze();
        maze[player_y][player_x] = PATH;

        if (player_x == EXIT_X && player_y == EXIT_Y) {
            printf("\n！\n");
            break;
        }

        handle_input();
    }

    return 0;
}

void initialize_maze() {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            maze[y][x] = WALL;
        }
    }
}

int is_valid(int x, int y) {
    return x >= 0 && x < WIDTH && y >= 0 && y < HEIGHT;
}

// dfs
void generate_maze_recursive(int x, int y) {
    maze[y][x] = PATH;

    int directions[4] = {0, 1, 2, 3};
    int dx[] = {0, 2, 0, -2};
    int dy[] = {-2, 0, 2, 0};

    for (int i = 0; i < 4; i++) {
        int j = rand() % 4;
        int temp = directions[i];
        directions[i] = directions[j];
        directions[j] = temp;
    }

    for (int i = 0; i < 4; i++) {
        int nx = x + dx[directions[i]];
        int ny = y + dy[directions[i]];

        if (is_valid(nx, ny) && maze[ny][nx] == WALL) {
            maze[y + dy[directions[i]] / 2][x + dx[directions[i]] / 2] = PATH;
            generate_maze_recursive(nx, ny);
        }
    }
}

void display_maze() {
    clear_screen();
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            switch (maze[y][x]) {
                case WALL:   printf("##");
                             break;
                case PATH:   printf("  ");
                             break;
                case PLAYER: printf(" P");
                             break;
                case EXIT:   printf(" E");
                             break;
            }
        }
        printf("\n");
    }
    printf("\nwasd/箭头移动，出口为E\n");
    printf("按q退出\n");
    printf("你在(%d, %d)\n", player_x, player_y);
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

    int next_x = player_x;
    int next_y = player_y;

    switch (key) {
        case 'w': case 'W': next_y--; break;
        case 's': case 'S': next_y++; break;
        case 'a': case 'A': next_x--; break;
        case 'd': case 'D': next_x++; break;
        case 'q': case 'Q':
            printf("\n游戏退出\n");
            exit(0);
        default:
            return;
    }

    if (is_valid(next_x, next_y) && maze[next_y][next_x] != WALL) {
        player_x = next_x;
        player_y = next_y;
    }
}