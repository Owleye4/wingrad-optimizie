
#include <stdio.h>
const float GB = 1024*1024*1024;
int main(void) {
	int l, C, H, W, K, N; 
  	scanf("%d", &l);
	int i = 0;
	while(l--) {
		printf("\n****************************LAYER %d***********************************\n", i++);
		scanf("%d%d%d%d%d", &C, &H, &W, &K, &N);
		const int outHeight = H - 2;
		const int outWidth = W - 2;
		const int sizeI = H * W;
		const int sizeF = 3 * 3;
		const int sizeO = outHeight * outWidth;

		printf("\n\n float* u_arr = (float*)malloc(K * C * sizeof(float*) * 16)");
			printf("\n Allocates memory: %f GB;", K * C * sizeof(float*) * 16 / GB);

		printf("\n\n float *out_ref = (float *)malloc(sizeof(float) * N * K * sizeO )");
			printf("\n Allocates memory: %f GB;", sizeof(float) * N * K * sizeO / GB);

		printf("\n\n image = (float *)aligned_alloc(64, sizeof(float) * batch * C * sizeI)");
			printf("\n Allocates memory: %f GB;", sizeof(float) * N * C * sizeI  / GB);

		printf("\n\n filter = (float *)aligned_alloc(64, sizeof(float) * K * C * sizeF)");
			printf("\n Allocates memory: %f GB;", sizeof(float) * K * C * sizeF  / GB);

		printf("\n\n out = (float *)aligned_alloc(64, sizeof(float) * batch * K * sizeO)");
			printf("\n Allocates memory: %f GB;\n", sizeof(float) * N * K * sizeO  / GB);
			
	}
}