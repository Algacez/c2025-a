#include <stdio.h>
#include <stdlib.h>
#include <string.h>

long long mod_pow(long long base, long long exp, long long mod) {
    long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

long long gcd(long long a, long long b) {
    while (b != 0) {
        long long t = b;
        b = a % b;
        a = t;
    }
    return a;
}

long long egcd(long long a, long long b, long long *x, long long *y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    long long x1, y1;
    long long g = egcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

long long mod_inverse(long long e, long long phi) {
    long long x, y;
    long long g = egcd(e, phi, &x, &y);
    if (g != 1) {
        return -1;
    }
    long long res = x % phi;
    if (res < 0) res += phi;
    return res;
}

void generate_keys(long long *e_out, long long *d_out, long long *n_out) {
    long long p = 61;
    long long q = 53;
    long long n = p * q;
    long long phi = (p - 1) * (q - 1);
    long long e = 17;
    if (gcd(e, phi) != 1) {
        for (e = 3; e < phi; e += 2) {
            if (gcd(e, phi) == 1) break;
        }
    }
    long long d = mod_inverse(e, phi);
    if (d == -1) {
        printf("无法生成私钥，请检查 p,q 是否为素数。\n");
        exit(1);
    }
    *e_out = e;
    *d_out = d;
    *n_out = n;
}

void encrypt(const char *plaintext, long long e, long long n, long long *cipher, int *out_len) {
    int len = strlen(plaintext);
    for (int i = 0; i < len; i++) {
        unsigned char m = (unsigned char)plaintext[i];
        cipher[i] = mod_pow((long long)m, e, n);
    }
    *out_len = len;
}

void decrypt(const long long *cipher, int len, long long d, long long n, char *out) {
    for (int i = 0; i < len; i++) {
        long long m = mod_pow(cipher[i], d, n);
        out[i] = (char)m;
    }
    out[len] = '\0';
}

int main(void) {
    char plaintext[1024];
    printf("加密的字符串：\n");
    if (fgets(plaintext, sizeof(plaintext), stdin) == NULL) {
        printf("读取失败。\n");
        return 1;
    }
    size_t ln = strlen(plaintext);
    if (ln > 0 && plaintext[ln - 1] == '\n') {
        plaintext[ln - 1] = '\0';
    }

    long long e, d, n;
    generate_keys(&e, &d, &n);

    printf("公钥 (e, n) = (%lld, %lld)\n私钥 d = %lld\n", e, n, d);
    printf("\n");

    long long cipher[1024];
    int cipher_len = 0;
    encrypt(plaintext, e, n, cipher, &cipher_len);

    printf("加密后：\n");
    for (int i = 0; i < cipher_len; i++) {
        printf("%lld ", cipher[i]);
    }
    printf("\n");
    printf("\n");

    char recovered[1024];
    decrypt(cipher, cipher_len, d, n, recovered);
    printf("解密后：\n%s", recovered);

    return 0;
}