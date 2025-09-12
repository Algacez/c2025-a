#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main() {
    char word[] = "A";
    int width = 60;      // 控制台一行的显示宽度
    int pos = 0;         // 初始位置
    int dir = 1;         // 方向：1 向右，-1 向左

    while (1) {
        // 清屏，使用 ANSI 转义序列
        printf("\033[2J\033[H");

        // 打印空格 + 单词
        for (int i = 0; i < pos; i++) {
            putchar(' ');
        }
        printf("%s\n", word);
        fflush(stdout);

        usleep(80000);  // 80 毫秒

        pos += dir;
        // 检查边界
        if (pos <= 0) {
            pos = 0;
            dir = 1;
        } else if (pos >= width - (int)strlen(word)) {
            pos = width - (int)strlen(word);
            dir = -1;
        }
    }
}