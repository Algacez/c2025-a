#include <stdio.h>
#include <time.h>

#define MAXN 1000

int main() {
    int prime[MAXN];
    int isComposite[MAXN] = {0};
    int count = 0;

    clock_t start_time, end_time;
    double time_used;

    start_time = clock();

    for (int i = 2; i <= MAXN; i++) {
        if (!isComposite[i]) {
            prime[count++] = i;
            printf("%d\n", i);
        }
        for (int j = 0; j < count && i * prime[j] <= MAXN; j++) {
            isComposite[i * prime[j]] = 1;
            if (i % prime[j] == 0)
                break;
        }
    }

    printf("2-1000有 %d 个质数", count);

    end_time = clock();
    time_used = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\n用时 %f 秒\n", time_used);

    return 0;
}
