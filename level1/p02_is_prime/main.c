#include <stdio.h>

void checkPrimeNumber();

int main()
{
    checkPrimeNumber();
    return 0;
}

void checkPrimeNumber()
{
    int n = 0, i = 0, flag = 0;

    printf("输入一个正整数: ");
    scanf("%d", &n);

    for(i=2; i <= n/2; ++i)
    {
        if(n%i == 0)
        {
            flag = 1;
            break;
        }
    }
    if (flag == 1)
        printf("%d 不是质数。", n);
    else
        printf("%d 是个质数。", n);
}