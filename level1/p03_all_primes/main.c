#include <printf.h>
#include <time.h>

int main() {
    int start = 2;
    int end = 1000;
    int count = 0;

    clock_t start_time, end_time;
    double time_used;

    start_time = clock();

    for (int i = start; i <= end; ++i)
    {
        int isPrime = 1;

        for (int j = 2; j * j <= i; ++j)
        {
            if (i % j == 0)
            {
                isPrime = 0;
                break;
            }
        }

        if (isPrime == 1)
        {
            printf("%d\n", i);
            count++;
        }
    }
    printf("2-1000有 %d 个质数", count);

    end_time = clock();
    time_used = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\n用时 %f 秒\n", time_used);

    return 0;
}