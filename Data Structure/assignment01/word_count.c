#include <stdio.h>
#include <stdlib.h> // malloc, realloc, free, qsort
#include <string.h> // strdup, strcmp, memmove

#define SORT_BY_WORD 0 // 단어 순 정렬
#define SORT_BY_FREQ 1 // 빈도 순 정렬

// 구조체 선언
// 단어 구조체
typedef struct
{
	char *word; // 단어
	int freq;	// 빈도
} tWord;

// 사전(dictionary) 구조체
typedef struct
{
	int len;	  // 배열에 저장된 단어의 수
	int capacity; // 배열의 용량 (배열에 저장 가능한 단어의 수)
	tWord *data;  // 단어 구조체 배열에 대한 포인터
} tWordDic;

////////////////////////////////////////////////////////////////////////////////
// 함수 원형 선언(declaration)

// 단어를 사전에 저장
// 새로 등장한 단어는 사전에 추가
// 이미 사전에 존재하는(저장된) 단어는 해당 단어의 빈도를 갱신(update)
// capacity는 1000으로부터 시작하여 1000씩 증가 (1000, 2000, 3000, ...)
void word_count(FILE *fp, tWordDic *dic);

// 사전을 화면에 출력 ("단어\t빈도" 형식)
void print_dic(tWordDic *dic);

// 사전에 할당된 메모리를 해제
void destroy_dic(tWordDic *dic);

// qsort를 위한 비교 함수
// 정렬 기준 : 단어
int compare_by_word(const void *n1, const void *n2);

// 정렬 기준 : 빈도 내림차순(1순위), 단어(2순위)
int compare_by_freq(const void *n1, const void *n2);

////////////////////////////////////////////////////////////////////////////////
// 이진탐색 함수
// found : key가 발견되는 경우 1, key가 발견되지 않는 경우 0
// return value: key가 발견되는 경우, 배열의 인덱스
//				key가 발견되지 않는 경우, key가 삽입되어야 할 배열의 인덱스
int binary_search(const void *key, const void *base, size_t nmemb, size_t size,
				  int (*compare)(const void *, const void *), int *found);

////////////////////////////////////////////////////////////////////////////////
// 함수 정의 (definition)
/////////////-------위 함수들 구현 여기에 ------------------///////////////////

// 단어를 사전에 저장하는 함수 구현
void word_count(FILE *fp, tWordDic *dic)
{
	char buffer[256];
	while (fscanf(fp, "%255s", buffer) == 1)
	{
		// 이진탐색을 위해 임시 tWord 생성 (word 필드만 사용)
		tWord temp;
		temp.word = buffer;
		temp.freq = 0;

		int found;
		// 사전은 단어순으로 정렬되어 있으므로 이진탐색 사용
		int index = binary_search(&temp, dic->data, dic->len, sizeof(tWord), compare_by_word, &found);

		if (found)
		{
			// 이미 존재하는 단어이면 빈도 증가
			dic->data[index].freq++;
		}
		else
		{
			// 새 단어이면 해당 위치에 삽입
			if (dic->len == dic->capacity)
			{
				dic->capacity += 1000;
				dic->data = (tWord *)realloc(dic->data, dic->capacity * sizeof(tWord));
			}
			// 삽입 위치 이후 요소들을 한 칸씩 이동
			if (index < dic->len)
			{
				memmove(&dic->data[index + 1], &dic->data[index], (dic->len - index) * sizeof(tWord));
			}
			// 새 단어의 메모리 할당 및 초기화
			dic->data[index].word = strdup(buffer);
			dic->data[index].freq = 1;
			dic->len++;
		}
	}
}

// 사전을 화면에 출력 ("단어\t빈도" 형식)
void print_dic(tWordDic *dic)
{
	for (int i = 0; i < dic->len; i++)
	{
		printf("%s\t%d\n", dic->data[i].word, dic->data[i].freq);
	}
}

// 사전에 할당된 메모리를 해제
void destroy_dic(tWordDic *dic)
{
	for (int i = 0; i < dic->len; i++)
	{
		free(dic->data[i].word);
	}
	free(dic->data);
	free(dic);
}

// compare_by_word 함수 구현 (단어 오름차순)
int compare_by_word(const void *n1, const void *n2)
{
	const tWord *w1 = (const tWord *)n1;
	const tWord *w2 = (const tWord *)n2;
	return strcmp(w1->word, w2->word);
}

// compare_by_freq 함수 구현 (빈도 내림차순, 빈도 같으면 단어 오름차순)
int compare_by_freq(const void *n1, const void *n2)
{
	const tWord *w1 = (const tWord *)n1;
	const tWord *w2 = (const tWord *)n2;
	if (w2->freq != w1->freq)
		return w2->freq - w1->freq;
	return strcmp(w1->word, w2->word);
}

// 이진탐색 함수 구현
int binary_search(const void *key, const void *base, size_t nmemb, size_t size,
				  int (*compare)(const void *, const void *), int *found)
{
	int low = 0, high = (int)nmemb - 1;
	int mid, cmp;
	while (low <= high)
	{
		mid = (low + high) / 2;
		const void *elem = (const char *)base + mid * size;
		cmp = compare(key, elem);
		if (cmp == 0)
		{
			*found = 1;
			return mid;
		}
		else if (cmp < 0)
		{
			high = mid - 1;
		}
		else
		{
			low = mid + 1;
		}
	}
	*found = 0;
	return low;
}

// 사전을 초기화 (빈 사전을 생성, 메모리 할당)
// len를 0으로, capacity를 1000으로 초기화
// return : 구조체 포인터
tWordDic *create_dic(void)
{
	tWordDic *dic = (tWordDic *)malloc(sizeof(tWordDic));

	dic->len = 0;
	dic->capacity = 1000;
	dic->data = (tWord *)malloc(dic->capacity * sizeof(tWord));

	return dic;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	tWordDic *dic;
	int option;
	FILE *fp;

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

	// 사전 초기화
	dic = create_dic();

	// 입력 파일 열기
	if ((fp = fopen(argv[2], "r")) == NULL)
	{
		fprintf(stderr, "cannot open file : %s\n", argv[2]);
		return 1;
	}

	// 입력 파일로부터 단어와 빈도를 사전에 저장
	word_count(fp, dic);

	fclose(fp);

	// 정렬 (빈도 내림차순, 빈도가 같은 경우 단어순)
	if (option == SORT_BY_FREQ)
	{
		qsort(dic->data, dic->len, sizeof(tWord), compare_by_freq);
	}

	// 사전을 화면에 출력
	print_dic(dic);

	// 사전 메모리 해제
	destroy_dic(dic);

	return 0;
}
