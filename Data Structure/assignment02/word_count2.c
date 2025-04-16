#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SORT_BY_WORD 0
#define SORT_BY_FREQ 1

typedef struct
{
    char *word;
    int freq;
} tWord;

typedef struct node
{
    tWord *dataPtr;
    struct node *link;
    struct node *link2;
} NODE;

typedef struct
{
    int count;
    NODE *head;
    NODE *head2;
} LIST;

LIST *createList(void)
{
    LIST *list = (LIST *)malloc(sizeof(LIST));
    if (!list)
        return NULL;
    list->count = 0;
    list->head = NULL;
    list->head2 = NULL;
    return list;
}

void destroyWord(tWord *pNode)
{
    if (pNode)
    {
        free(pNode->word);
        free(pNode);
    }
}

void destroyList(LIST *pList)
{
    NODE *current = pList->head;
    while (current)
    {
        NODE *temp = current;
        current = current->link;
        destroyWord(temp->dataPtr);
        free(temp);
    }
    free(pList);
}

static int _search(LIST *pList, NODE **pPre, NODE **pLoc, tWord *pArgu)
{
    *pPre = NULL;
    *pLoc = pList->head;

    while (*pLoc && strcmp((*pLoc)->dataPtr->word, pArgu->word) < 0)
    {
        *pPre = *pLoc;
        *pLoc = (*pLoc)->link;
    }

    if (*pLoc && strcmp((*pLoc)->dataPtr->word, pArgu->word) == 0)
        return 1;

    return 0;
}

static int _insert(LIST *pList, NODE *pPre, tWord *dataInPtr)
{
    NODE *newNode = (NODE *)malloc(sizeof(NODE));
    if (!newNode)
        return 0;

    newNode->dataPtr = dataInPtr;
    newNode->link = NULL;
    newNode->link2 = NULL;

    if (!pPre)
    {
        newNode->link = pList->head;
        pList->head = newNode;
    }
    else
    {
        newNode->link = pPre->link;
        pPre->link = newNode;
    }
    pList->count++;
    return 1;
}

int addNode(LIST *pList, tWord *dataInPtr)
{
    NODE *pPre, *pLoc;
    int found = _search(pList, &pPre, &pLoc, dataInPtr);

    if (found)
    {
        pLoc->dataPtr->freq++;
        return 2;
    }
    else
    {
        return _insert(pList, pPre, dataInPtr) ? 1 : 0;
    }
}

static int _search_by_freq(LIST *pList, NODE **pPre, NODE **pLoc, tWord *pArgu)
{
    *pPre = NULL;
    *pLoc = pList->head2;

    while (*pLoc && ((*pLoc)->dataPtr->freq > pArgu->freq ||
                     ((*pLoc)->dataPtr->freq == pArgu->freq &&
                      strcmp((*pLoc)->dataPtr->word, pArgu->word) < 0)))
    {
        *pPre = *pLoc;
        *pLoc = (*pLoc)->link2;
    }

    return 0;
}

static void _link_by_freq(LIST *pList, NODE *pPre, NODE *pLoc)
{
    if (!pPre)
    {
        pLoc->link2 = pList->head2;
        pList->head2 = pLoc;
    }
    else
    {
        pLoc->link2 = pPre->link2;
        pPre->link2 = pLoc;
    }
}

void connect_by_frequency(LIST *list)
{
    list->head2 = NULL;
    NODE *cur = list->head;

    while (cur)
    {
        NODE *pPre, *pLoc;
        _search_by_freq(list, &pPre, &pLoc, cur->dataPtr);
        _link_by_freq(list, pPre, cur);
        cur = cur->link;
    }
}

void print_dic(LIST *pList)
{
    NODE *cur = pList->head;
    while (cur)
    {
        printf("%s\t%d\n", cur->dataPtr->word, cur->dataPtr->freq);
        cur = cur->link;
    }
}

void print_dic_by_freq(LIST *pList)
{
    NODE *cur = pList->head2;
    while (cur)
    {
        printf("%s\t%d\n", cur->dataPtr->word, cur->dataPtr->freq);
        cur = cur->link2;
    }
}

tWord *createWord(char *word)
{
    tWord *p = (tWord *)malloc(sizeof(tWord));
    if (!p)
        return NULL;
    p->word = strdup(word);
    if (!p->word)
    {
        free(p);
        return NULL;
    }
    p->freq = 1;
    return p;
}

int compare_by_word(const void *n1, const void *n2)
{
    tWord *p1 = (tWord *)n1;
    tWord *p2 = (tWord *)n2;
    return strcmp(p1->word, p2->word);
}

int compare_by_freq(const void *n1, const void *n2)
{
    tWord *p1 = (tWord *)n1;
    tWord *p2 = (tWord *)n2;
    int ret = p2->freq - p1->freq;
    if (ret != 0)
        return ret;
    return strcmp(p1->word, p2->word);
}

int main(int argc, char **argv)
{
    LIST *list;
    int option;
    FILE *fp;
    char word[100];
    tWord *pWord;
    int ret;

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s option FILE\n\n", argv[0]);
        fprintf(stderr, "option\n\t-w\t\tsort by word\n\t-f\t\tsort by frequency\n");
        return 1;
    }

    if (strcmp(argv[1], "-w") == 0)
        option = SORT_BY_WORD;
    else if (strcmp(argv[1], "-f") == 0)
        option = SORT_BY_FREQ;
    else
    {
        fprintf(stderr, "unknown option : %s\n", argv[1]);
        return 1;
    }

    list = createList();
    if (!list)
    {
        printf("Cannot create list\n");
        return 100;
    }

    if ((fp = fopen(argv[2], "r")) == NULL)
    {
        fprintf(stderr, "cannot open file : %s\n", argv[2]);
        return 2;
    }

    while (fscanf(fp, "%s", word) != EOF)
    {
        pWord = createWord(word);
        if (!pWord)
            continue;

        ret = addNode(list, pWord);

        if (ret == 0 || ret == 2)
        {
            destroyWord(pWord);
        }
    }

    fclose(fp);

    if (option == SORT_BY_WORD)
    {
        print_dic(list);
    }
    else
    {
        connect_by_frequency(list);
        print_dic_by_freq(list);
    }

    destroyList(list);
    return 0;
}
