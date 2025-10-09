cmake_minimum_required(VERSION 4.0)
project(c2025_challenge)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(P10_SOURCES
        level1/p10_warehouse/main.c
        level1/p10_warehouse/cJSON.c
)

add_executable(p01_running_letter level1/p01_running_letter/main.c)

add_executable(p02_is_prime  level1/p02_is_prime/main.c)

add_executable(p03_all_primes  level1/p03_all_primes/main.c)

add_executable(p04_goldbach  level1/p04_goldbach/main.c)

add_executable(p05_encrypt_decrypt  level1/p05_encrypt_decrypt/main.c)

add_executable(p06_hanoi  level1/p06_hanoi/main.c
        level1/p06_hanoi/hanoi.c
        level1/p06_hanoi/hanoi.h
)

add_executable(p07_maze  level1/p07_maze/main.c)

add_executable(p08_push_boxes  level1/p08_push_boxes/main.c)

add_executable(p09_linked_list  level1/p09_linked_list/main.c)

add_executable(p10_warehouse
        ${P10_SOURCES}
)
