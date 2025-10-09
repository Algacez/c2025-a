#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"

#define MAX_ITEMS 100
#define MAX_MODEL_LEN 50
#define FILENAME "inventory.json"

typedef struct {
    char model[MAX_MODEL_LEN];
    int quantity;
} Product;

typedef struct {
    Product products[MAX_ITEMS];
    int count;
} Inventory;

Inventory inventory = {0};

void loadInventory();
void saveInventory();
void displayInventory();
void addStock();
void removeStock();
void showMenu();
int findProduct(const char *model);

int main() {
    int choice;

    printf("欢迎使用进销存管理系统\n");
    loadInventory();

    while (1) {
        showMenu();
        printf("请选择操作: ");

        if (scanf("%d", &choice) != 1) {
            while (getchar() != '\n');
            printf("输入无效，请输入数字\n");
            continue;
        }
        while (getchar() != '\n');

        switch (choice) {
            case 1:
                displayInventory();
                break;
            case 2:
                addStock();
                break;
            case 3:
                removeStock();
                break;
            case 4:
                printf("正在保存数据并退出...\n");
                saveInventory();
                printf("感谢使用\n");
                return 0;
            default:
                printf("无效选择，请重新输入\n");
        }
        printf("\n");
    }
    return 0;
}

void showMenu() {
    printf("1. 显示存货列表\n");
    printf("2. 入库\n");
    printf("3. 出库\n");
    printf("4. 退出程序\n");
}

int findProduct(const char *model) {
    for (int i = 0; i < inventory.count; i++) {
        if (strcmp(inventory.products[i].model, model) == 0) {
            return i;
        }
    }
    return -1;
}

void displayInventory() {
    printf("\n存货列表");
    if (inventory.count == 0) {
        printf("库存为空\n");
    } else {
        printf("%-20s %s\n", "型号", "数量");
        for (int i = 0; i < inventory.count; i++) {
            printf("%-20s %d\n",
                   inventory.products[i].model,
                   inventory.products[i].quantity);
        }
    }
}

void addStock() {
    char model[MAX_MODEL_LEN];
    int quantity;

    printf("\n入库\n");
    printf("请输入货物型号: ");
    if (scanf("%49s", model) != 1) {
        printf("输入错误\n");
        while (getchar() != '\n');
        return;
    }

    printf("请输入入库数量: ");
    if (scanf("%d", &quantity) != 1 || quantity <= 0) {
        printf("数量无效\n");
        while (getchar() != '\n');
        return;
    }
    while (getchar() != '\n');

    int index = findProduct(model);
    if (index != -1) {
        inventory.products[index].quantity += quantity;
        printf("入库成功型号 [%s] 当前库存: %d\n",
               model, inventory.products[index].quantity);
    } else {
        if (inventory.count >= MAX_ITEMS) {
            printf("库存已满，无法添加新货物\n");
            return;
        }
        strcpy(inventory.products[inventory.count].model, model);
        inventory.products[inventory.count].quantity = quantity;
        inventory.count++;
        printf("入库成功新增型号 [%s]，数量: %d\n", model, quantity);
    }
}

void removeStock() {
    char model[MAX_MODEL_LEN];
    int quantity;

    printf("\n出库\n");
    printf("请输入货物型号: ");
    if (scanf("%49s", model) != 1) {
        printf("输入错误\n");
        while (getchar() != '\n');
        return;
    }

    printf("请输入出库数量: ");
    if (scanf("%d", &quantity) != 1 || quantity <= 0) {
        printf("数量无效\n");
        while (getchar() != '\n');
        return;
    }
    while (getchar() != '\n');

    int index = findProduct(model);
    if (index == -1) {
        printf("错误：型号 [%s] 不存在\n", model);
        return;
    }

    if (inventory.products[index].quantity < quantity) {
        printf("错误：库存不足当前库存: %d\n",
               inventory.products[index].quantity);
        return;
    }

    inventory.products[index].quantity -= quantity;
    printf("出库成功型号 [%s] 剩余库存: %d\n",
           model, inventory.products[index].quantity);

    if (inventory.products[index].quantity == 0) {
        printf("该货物库存为0，已从列表中移除。\n");
        for (int i = index; i < inventory.count - 1; i++) {
            inventory.products[i] = inventory.products[i + 1];
        }
        inventory.count--;
    }
}

void loadInventory() {
    FILE *file = fopen(FILENAME, "r");
    if (!file) {
        printf("未找到库存文件，将创建新的库存。\n");
        inventory.count = 0;
        return;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *data = (char *)malloc(length + 1);
    if (!data) {
        printf("内存分配失败\n");
        fclose(file);
        return;
    }

    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    cJSON *json = cJSON_Parse(data);
    free(data);

    if (!json) {
        printf("JSON解析失败\n");
        inventory.count = 0;
        return;
    }

    cJSON *products = cJSON_GetObjectItem(json, "products");
    if (!products || !cJSON_IsArray(products)) {
        printf("JSON格式错误\n");
        cJSON_Delete(json);
        inventory.count = 0;
        return;
    }

    inventory.count = 0;
    cJSON *item = NULL;
    cJSON_ArrayForEach(item, products) {
        if (inventory.count >= MAX_ITEMS) break;

        cJSON *model = cJSON_GetObjectItem(item, "model");
        cJSON *quantity = cJSON_GetObjectItem(item, "quantity");

        if (cJSON_IsString(model) && cJSON_IsNumber(quantity)) {
            strncpy(inventory.products[inventory.count].model,
                    model->valuestring, MAX_MODEL_LEN - 1);
            inventory.products[inventory.count].model[MAX_MODEL_LEN - 1] = '\0';
            inventory.products[inventory.count].quantity = quantity->valueint;
            inventory.count++;
        }
    }

    cJSON_Delete(json);
    printf("成功加载 %d 条库存记录。\n", inventory.count);
}

void saveInventory() {
    cJSON *json = cJSON_CreateObject();
    cJSON *products = cJSON_CreateArray();

    for (int i = 0; i < inventory.count; i++) {
        cJSON *item = cJSON_CreateObject();
        cJSON_AddStringToObject(item, "model", inventory.products[i].model);
        cJSON_AddNumberToObject(item, "quantity", inventory.products[i].quantity);
        cJSON_AddItemToArray(products, item);
    }

    cJSON_AddItemToObject(json, "products", products);

    char *jsonString = cJSON_Print(json);

    FILE *file = fopen(FILENAME, "w");
    if (!file) {
        printf("无法打开文件保存\n");
        cJSON_Delete(json);
        free(jsonString);
        return;
    }

    fprintf(file, "%s", jsonString);
    fclose(file);

    cJSON_Delete(json);
    free(jsonString);

    printf("库存数据已保存到 %s\n", FILENAME);
}