#include <stdio.h>


int primes[50];
int iscomposite[101] = {0};
int count = 0;

void generate_primes() {
    for (int i = 2; i <= 100; i++)
    {
        if (!iscomposite[i])
        {
            primes[count++] = i;
            printf("%d\n", i);
        }
        for (int j = 0; j <= count && primes[j] * i <= 100; j++)
        {
            iscomposite[primes[j] * i] = 1;
            if (i % primes[j] == 0)
                break;
        }
    }
}

int check_true(void)
{
    for (int even = 4; even <= 100; even += 2){
        int found = 0;
        for (int a = 0; a < count; a++)
        {
            if (primes[a] > even / 2) {
                break; // 优化：如果 a 本身就超过了 even 的一半，b 只能比它更大，和肯定超了
            }
            for (int b = a; b < count; b++)
            {
                int check = primes[a] + primes[b];
                if (even == check)
                {
                    printf("%d 成立\n", even);
                    found = 1;
                    break;
                }
                if (check > even)
                {
                    break;
                }
            }
            if (found)
            {
                break;
            }
        }
        if (!found) {
            printf("哥德巴赫猜想不成立！偶数 %d 无法分解为两个素数之和。\n", even);
            return 1;
        }
    }
    return 0;
}

int main()
{
    generate_primes();
    return check_true();
}