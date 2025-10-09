//
// Created by Rock Zhang on 2025/9/22.
//
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int data;
    struct Node* next; //指向同一类型(struct Node)结构体的指针
} Node;

typedef struct {
    Node* head;
    int length;
} LinkedList;

void initList(LinkedList*  list)
{
    list -> head = NULL;
    list -> length = 0;
}

void insert(LinkedList* list, int data)
{
    Node* newnode = (Node*)malloc(sizeof(Node));
    newnode -> data = data;
    newnode -> next = list -> head;  // 新节点就连接到了链表的现有部分之前
    list -> head = newnode;  // 新节点成为了链表的新头部
    list -> length++;
}

void append(LinkedList* list, int data)
{
    Node* newnode = (Node*)malloc(sizeof(Node));
    newnode -> data = data;
    newnode -> next = NULL;
    if (list -> head == NULL)
    {
        list -> head = newnode;
    }
    else
    {
        Node* current = list -> head;
        while (current -> next != NULL)
        {
            current = current -> next;
        }
        current -> next = newnode;
    }
}

int search(LinkedList* list, int data)
{
    Node* current = list -> head;
    int index = 0;
    while (current != NULL)
    {
        if (current -> data == data)
        {
            printf("你搜索的数 %d 的索引是 %d\n", data, index);
        }
        current = current -> next;
        index++;
    }
    return -1;
}

void reverse(LinkedList* list)
{
    if (list->head == NULL || list->head->next == NULL) {
        // 如果链表为空或只有一个节点，无需反转，直接返回
        return;
    }
    Node* prev_node = NULL;      // 指向前一个节点
    Node* current = list->head; // 指向当前节点
    Node* next_node = NULL;     // 临时指针，用于保存下一个节点
    // 遍历链表
    while (current != NULL) {
        // 1. 保存当前节点的下一个节点
        next_node = current->next;

        // 2. 将当前节点的next指针指向前一个节点（实现反转）
        current->next = prev_node;

        // 3. prev和current指针向后移动，为下一次反转做准备
        prev_node = current;
        current = next_node;
    }
    // 循环结束后，prev指向原链表的最后一个节点，即新链表的头节点
    list->head = prev_node;
}

void printList(LinkedList* list)
{
    Node* current = list -> head;
    while (current != NULL)
    {
        printf("%d ", current -> data);
        current = current -> next;
    }
    printf("\n");
}

int main()
{
    LinkedList list;
    initList(&list);
    insert(&list, 2);
    insert(&list, 1);
    append(&list, 5);
    insert(&list, 3);
    insert(&list, 5);
    append(&list, 4);
    printList(&list);
    reverse(&list);
    search(&list, 5);
    return 0;
}