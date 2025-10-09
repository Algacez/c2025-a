#include <stdio.h>
#include "hanoi.h"

int main()
{
    int n;
    scanf("%d", &n);
    hanoi(n, 'A', 'B', 'C');
}