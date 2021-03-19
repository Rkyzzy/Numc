#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, double low, double high) {
    srand(42);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
	if(mat==NULL||rows<=0||cols<=0){
		return -1;
	}
	*mat = (matrix*)malloc(sizeof(matrix));
	if(*mat==NULL){
		return -1;
	}
	(*mat)->rows = rows;
	(*mat)->cols = cols;
	(*mat)->data = (double*)calloc(rows*cols,sizeof(double));
	if((*mat)->data==NULL){
		return -1;
	}
	(*mat)->ref_cnt = 1;
	(*mat)->parent = NULL;
	return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
	if(mat==NULL||rows<=0||cols<=0){
		return -1;
	}
	*mat = (matrix*)malloc(sizeof(matrix));
	if(*mat==NULL){
		return -1;
	}
	(*mat)->data = (double*)(from->data+offset);
	(*mat)->parent = from;
	from->ref_cnt++;
	(*mat)->ref_cnt = 1;
	(*mat)->rows = rows;
	(*mat)->cols = cols;
	return 0;
}

/*
 * This function frees the matrix struct pointed to by `mat`. However, you need to make sure that
 * you only free the data if `mat` is not a slice and has no existing slices, or if `mat` is the
 * last existing slice of its parent matrix and its parent matrix has no other references.
 * You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
	if(mat!=NULL){
		if(mat->parent==NULL&&mat->ref_cnt==1){
			free(mat->data);
			free(mat);
		}
		else if(mat->parent!=NULL&&mat->parent->ref_cnt==2&&mat->parent->rows==-1){
			free(mat->parent);
			free(mat);
		}
		else{
			if(mat->parent==NULL&&mat->ref_cnt>1){
				free(mat->data);
				mat->rows = -1;
			}
			if(mat->parent!=NULL){
				mat->parent->ref_cnt--;
				free(mat);
			}
		}
	}
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
	return mat->data[row*(mat->cols)+col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
	mat->data[row*(mat->cols)+col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
	memset(mat->data,val,(mat->rows)*(mat->cols)*sizeof(double));
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
	if(mat1==NULL||mat1->data==NULL||mat2==NULL||mat2->data==NULL||mat1->rows<0||mat1->cols<0||mat2->rows<0||mat2->cols<0||result==NULL||mat1->rows!=mat2->rows||mat1->cols!=mat2->cols||result->rows!=mat1->rows||result->cols!=mat1->cols){
		return -1;
	}
	int total = (mat1->rows*(mat1->cols));
	int aligned = total - total %4;
        #pragma omp parallel for
	for(int i = 0; i<aligned;i+=4){
		//result->data[i] = mat1->data[i] + mat2->data[i];
		__m256d c0 = {0,0,0,0};
		c0 = _mm256_add_pd(_mm256_loadu_pd((mat1->data)+i),_mm256_loadu_pd((mat2->data)+i));
		_mm256_storeu_pd((result->data+i),c0);
	}
	for(int i = aligned; i < total; i++){
		result->data[i] = mat1->data[i] + mat2->data[i];
	}
	return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
	if(mat1==NULL||mat1->data==NULL||mat2==NULL||mat2->data==NULL||mat1->rows<0||mat1->cols<0||mat2->rows<0||mat2->cols<0||result==NULL||mat1->rows!=mat2->rows||mat1->cols!=mat2->cols||result->rows!=mat1->rows||result->cols!=mat1->cols){
		return -1;
	}
	//allocate_matrix(&result,mat1->rows,mat1->cols);
	int total = (mat1->rows*(mat1->cols));
        int aligned = total - total%4;
        #pragma omp parallel for
	for(int i = 0; i< aligned;i+=4){
		//result->data[i] = mat1->data[i] - mat2->data[i];
		__m256d c0 = {0,0,0,0};
		c0 = _mm256_sub_pd(_mm256_loadu_pd((mat1->data)+i),_mm256_loadu_pd((mat2->data)+i));
		_mm256_storeu_pd((result->data+i),c0);
	}
	for(int i = aligned; i<total;i++){
		result->data[i] = mat1->data[i] - mat2->data[i];
	}
	return 0;
}

void matrix_transposing(matrix *from, matrix* to){
        #pragma omp parallel for
	for(int i = 0; i<to->rows;i++){
		for(int j = 0; j<to->cols; j++){
			to->data[j+i*to->cols] = from->data[i+j*from->cols];
		}
	}
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
	if(mat1==NULL||mat2==NULL||result==NULL||mat1->cols!=mat2->rows||result->rows!=mat1->rows||result->cols!=mat2->cols){
		return -1;
	}
        /* 
	 //naive version
	//allocate_matrix(&result,mat1->rows,mat2->cols);
	for(int i = 0; i< mat1->rows; i++){
		for(int j = 0; j<mat2->cols; j++){
			double cij = 0;
			for(int k = 0; k< mat2->rows;k++){
				cij += (mat1->data[i*(mat1->cols)+k])*(mat2->data[j+k*(mat2->cols)]);
			}
			result->data[j+i*(result->cols)] = cij;
		}
	}
	*/
        
	//speedup version
	matrix* mat3temp;
	allocate_matrix(&mat3temp,mat2->cols,mat2->rows);
	matrix_transposing(mat2,mat3temp);
	#pragma omp parallel for
	for(int i = 0; i< mat1->rows;i++){
		for(int j = 0; j < mat3temp->rows; j++){
			result->data[j+i*(result->cols)] = 0;
			//try simd
			int total = mat1->cols;
			int aligned = total - total%4;
			__m256d c0 = {0,0,0,0};
			for(int k = 0; k < aligned; k+=4){
				c0 = _mm256_add_pd(
						_mm256_mul_pd(_mm256_loadu_pd(mat1->data+k+i*(mat1->cols)),
								_mm256_loadu_pd(mat3temp->data+k+j*(mat3temp->cols)))
						,c0 );
			}
			double* c0array = (double *)&c0;
			double tempval = 0;
			for(int k = 0; k <4; k++){
				tempval+= c0array[k];
			}
			result->data[j+i*(result->cols)] += tempval;
			for(int k = aligned; k < total; k++){
				result->data[j+i*(result->cols)] += mat1->data[k+i*(mat1->cols)]*mat3temp->data[k+j*(mat3temp->cols)];
			}
			
			//
			//for(int k = 0;k< mat1->cols; k++){
			//	result->data[j+i*(result->cols)] += mat1->data[k+i*(mat1->cols)] * mat3temp->data[k+j*(mat3temp->cols)];
			//}
		}
	}
	deallocate_matrix(mat3temp);
	
	return 0;
}




/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
	if(mat==NULL||result==NULL||mat->rows!=mat->cols||result->rows!=mat->rows||result->cols!=mat->cols){
		return -1;
	}
	//allocate_matrix(&result, mat->rows, mat->cols);
	/*
	matrix * temp = NULL;
	allocate_matrix(&temp,mat->rows,mat->cols);
	fill_matrix(temp,0);
	add_matrix(result,temp,mat);
        //#pragma omp parallel for
	for(int i = 1; i< pow; i++){
		fill_matrix(temp,0);
		add_matrix(temp,temp,result);
		mul_matrix(result,mat,temp);
	}
	deallocate_matrix(temp);
	*/

	if(pow==1){
		int total = mat->rows*(mat->cols);
		int aligned = total - total%4;
		for(int i = 0; i< total; i++){
			result->data[i] = mat->data[i];
		}
	}
	else if(pow==2){
		mul_matrix(result,mat,mat);
	}
	else{
		int flag = 0;
		if(pow%2==1){
			flag = 1;
		}
                pow/=2;
		matrix *mater;
		allocate_matrix(&mater,mat->rows,mat->cols);
		pow_matrix(mater,mat,pow);
		mul_matrix(result,mater,mater);
		deallocate_matrix(mater);
		if(flag==1){
			matrix *mattemp;
			allocate_matrix(&mattemp,mat->rows,mat->cols);
			for(int j = 0; j< mat->rows*(mat->cols); j++){
				mattemp->data[j] = result->data[j];
			}
                        mul_matrix(result,mat,mattemp);
			deallocate_matrix(mattemp);
		}
	}
	return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
	if(mat==NULL||result==NULL||result->rows!=mat->rows||result->cols!=mat->cols){
		return -1;
	}
	//allocate_matrix(&result,mat->rows,mat->cols);
	int total = mat->rows*(mat->cols);
	int aligned = total - total%4;
        #pragma omp parallel for
	for(int i = 0; i< aligned;i+=4){
		__m256d c0 = {0,0,0,0};
		c0 = _mm256_mul_pd (_mm256_set1_pd(-1) ,_mm256_loadu_pd((mat->data)+i));
		_mm256_storeu_pd((result->data)+i,c0);
	}
	for(int i = aligned;i<total;i++){
		result->data[i] = (-1)*(mat->data[i]);
	}
	return 0;
}

//abstool
__m256d abs_pd(__m256d in){
	return _mm256_max_pd(_mm256_sub_pd(_mm256_setzero_pd(),in),in);
}


/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
	if(mat == NULL||result==NULL||result->rows!=mat->rows||result->cols!=mat->cols){
		return -1;
	}
	//allocate_matrix(&result,mat->rows,mat->cols);
	int total = mat->rows*(mat->cols);
	int aligned = total - total%4;
        #pragma omp parallel for
	for(int i = 0; i< aligned; i+=4){
		_mm256_storeu_pd(result->data+i,abs_pd(_mm256_loadu_pd(mat->data+i)));
	}
	for(int i = aligned; i < total; i++){
		if(mat->data[i]<0){
			result->data[i] = (-1)*(mat->data[i]);
		}
		else{
			result->data[i] = mat->data[i];
		}
	}
	/*
	for(int i = 0; i< (mat->rows*(mat->cols));i++){
		if(mat->data[i]<0){
			result->data[i] = (-1)*(mat->data[i]);
		}
		else{
			result->data[i] = mat->data[i];
		}
	}
	*/
	return 0;
}

