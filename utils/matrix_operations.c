#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <math.h>
extern void dgetrf_(int* M, int *N, double* A, int* lda, int* IPIV, int* INFO);
extern void dgetri_(int* N, double* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
extern void zgetrf_(int* M, int *N, double complex* A, int* lda, int* IPIV, int* INFO);
extern void zgetri_(int* N, double complex* A, int* lda, int* IPIV, double* WORK, int* lwork, int* INFO);
int create_vector(double **matrix,int l)
{
    int i;
    (*matrix) = (double *) malloc(l * sizeof(double));
    if ( !(matrix) ) {
        return 0;
    }
    return 1;
}
void reshape(int m, int n, double** mat, double* cvec) {
	/*
		Turns a matrix (given by a double pointer) into its C vector format 
		(single vector, rowwise). The matrix "mat" needs to be an n*m matrix
		The vector "vec" is a vector of lenght nm
	*/
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++) 
			cvec[i*n+j] = mat[j][i];
}
void creshape(int m, int n, double complex** mat, double complex** cvec,int col) {
	/*
		Turns a matrix (given by a double pointer) into its C vector format 
		(single vector, rowwise). The matrix "mat" needs to be an n*m matrix
		The vector "vec" is a vector of lenght nm
	*/
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++) 
			cvec[i*n+j][col] = mat[j][i];
}
void creshapeB(int m, int n, double complex* cvec, double complex** mat) {
	/*
		Turns a matrix (given by a double pointer) into its C vector format 
		(single vector, rowwise). The matrix "mat" needs to be an n*m matrix
		The vector "vec" is a vector of lenght nm
	*/
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++)
			mat[j][i]=cvec[i*n+j];
}
double cMatMean(int n, int m, double complex** in){
	int i,j;
	double complex mean,sum;
	sum=0;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++)
			sum+=in[i][j];
	mean=sum/(n*m);
	return creal(mean);
}
void matCpy(int n, int m, double complex** in, double complex** out,double mult) {
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++)
			out[i][j]=mult*in[i][j];
}
void subsMat(int n, int m, CvMat* in, CvMat* out,double val) {
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++)
			out->data.fl[i*m+j]=in->data.fl[i*m+j]-val;
}
void subsMatrix(int n, int m, double complex** in, double complex** out,double val) {
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++)
			out[i][j]=creal(in[i][j])-val;
}
void matsAdd(int m, int n, double complex** a,double complex ** b, double complex** out) {
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++)
			out[j][i]=a[i][j]+b[i][j];
}
int get_matrix_order(char * filename, int * c,int * l)
{
	FILE * in_file;
	char * ans, * line, str1[256]; 
	int col_count, row_count, i, j;
	in_file = fopen(filename, "r");
	if ( !in_file ) {
		return 0;
	}

	col_count = 0;
	row_count = 0;
	if ( !fgets(str1, sizeof(str1),  in_file) ) {
		row_count = 1;
	} else {
		*l = 1;         // rows number
		*c = 1;         // cols number
		line = str1;
		while (!col_count) 
		{
			ans = strchr(line,'\t');
			if (!ans) {
				col_count = 1;
			} else {
				line = ans+1;
				*c = *c + 1;
			}
		}
	}
	while(!row_count)
	{
		if ( !fgets(str1, sizeof(str1),  in_file) ) {
			row_count = 1;
		} else {
			*l=*l+1;
		}
	}
	fclose(in_file);
	return 1;
}
void matXmat(double complex **a,double complex **b,double complex** c,int r1,int c1,int r2,int c2)
{
	double complex sum;
	//c1==r2
	for(int i=0; i<r1; ++i)
	{
	    for(int j=0; j<c2; ++j)
	    {	
	    	sum=0;
	        for(int k=0; k<c1; ++k)
	        {
	            sum+=a[i][k]*b[k][j];
	        }
	        c[i][j]=sum;
	    }
	}
}
void matXvec(double complex **a,double complex *b,double complex* c,int r1,int c1,int r2,int c2)
{
	double complex sum;
	for(int i=0; i<r1; ++i)
	{
	    for(int j=0; j<c2; ++j)
	    {	
	    	sum=0;
	        for(int k=0; k<c1; ++k)
	        {
	            sum+=a[i][k]*b[k];
	        }
	        c[i]=sum;
	    }
	}
}
void t1matXmat(double complex **a,double complex **b,double complex** c,int r1,int c1,int r2,int c2)
{
	double complex sum;
	for(int i=0; i<c1; ++i)
	{
	    for(int j=0; j<c2; ++j)
	    {	
	    	sum=0;
	        for(int k=0; k<r1; ++k)
	        {
	            sum+=a[k][i]*b[k][j];
	        }
	        c[i][j]=sum;
	    }
	}
}
void cmatrix_multiplication(double complex **a,double complex **b,double complex** c,int lines,int cols)
{
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			c[i][j]=b[i][j]*a[i][j];
		}
	}
}
void cmatrix_div(double complex **a,double complex **b,double complex** c,int lines,int cols)
{
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			c[i][j]=a[i][j]/creal(b[i][j]);
		}
	}
}
void dcmatrix_multiplication(double **a,double complex **b,double complex** c,int lines,int cols)
{
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			c[i][j]=b[i][j]*(a[i][j]);
		}
	}
}
void matrix_multiplication(double **a,CvMat* b,CvMat* c,int lines,int cols)
{
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			c->data.fl[i*c->cols+j]=(a[i][j])*(b->data.fl[i*b->cols+j]);
		}
	}
}
void vsum(int *a,int n,int* sum)
{
	*sum=0;
	for (int i = 0; i < n; ++i)
	{
		*sum+=a[i];
	}
}
void vector_multiplication(int *a,int *b,int* c,int n)
{
	for (int i = 0; i < n; ++i)
	{
		c[i]=a[i]*b[i];
	}
}
int create_matrix(double ***matrix, int c,int l)
{
	int i;
	(*matrix) = (double **) malloc(l * sizeof(double*));
	if ( !(*matrix) ) {
		return 0;
	}
	for (i = 0; i < l; i++)
	{
		(*matrix)[i]=(double *)malloc(c * sizeof(double));
		if (!(*matrix)[i])
		{
			return 0;
		}
	}
	return 1;
}
int create_cmatrix(double complex ***matrix, int c,int l)
{
	int i;
	(*matrix) = (double complex**) malloc(l * sizeof(double complex*));
	if ( !(*matrix) ) {
		return 0;
	}
	for (i = 0; i < l; i++)
	{
		(*matrix)[i]=(double complex*)malloc(c * sizeof(double complex));
		if (!(*matrix)[i])
		{
			return 0;
		}
	}
	return 1;
}
int read_matrix_from_file(char * filename, double *** matrix, int c,int r)
{
	FILE * in_file;
	char * ans, * line, str1[256], no[80],*ptr; 
	int i, j, k, l;
	double n;
	ptr=NULL;
	in_file = fopen(filename, "r");
	if ( !in_file ) {
		return 0;
	}

	for (i=0; i<r; i++) {
		if ( !fgets(str1, sizeof(str1),  in_file) ) {
			return 0;
		} else {
			line = str1;
			for (j=0; j<c-1; j++) {
				memset(no, '\0', strlen(no));
				ans = strchr(line,'\t');
				if ( ans ) {
					k = strlen(ans);
					l = strlen(line);
					strncpy(no, line, l-k);
					n=strtod(no,&ptr);
					(*matrix)[i][j] = n;
					line = ans+1;
				}
			}
			memset(no, '\0', strlen(no));
			ans = strchr(line,'\n');
			k = strlen(ans);
			l = strlen(line);
			strncpy(no, line, l-k);
			n=strtod(no,&ptr);
			(*matrix)[i][j] = n;      
		}
	}  
	fclose(in_file);
	return 1;
}
void print_matrix_data(double** matrix,int c,int l)
{
	int i,j;
	for (i = 0; i < l; i++)
	{
		for (j = 0; j < c; j++)
		{	
			printf("%f ",matrix[i][j]);
		}
		printf("\n");
	}
}
void add_matrixes(int** a,int** b,int*** c,int cols,int lines)
{
	int i,j;
	for (i = 0; i < lines; i++)
	{
		for (j = 0; j < cols; j++)
		{	
			(*c)[i][j] = a[i][j] + b[i][j];
		}
	}
}
int write_matrix_into_file(char* file,int** matrix,int c,int l)
{
	FILE *fp;
	int i,j;
	fp=fopen(file,"w");
	if(!fp)
		return 0;
	for (i = 0; i < l; i++)
	{
		for (j = 0; j < c; j++)
		{	
			fprintf(fp,"%d\t",matrix[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);	
	return 1;
}
void reshapeF(int m, int n, double** mat, double* cvec) {
	/*
		Turns a matrix (given by a double pointer) into its C vector format 
		(single vector, rowwise). The matrix "mat" needs to be an n*m matrix
		The vector "vec" is a vector of lenght nm
	*/
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++) 
			cvec[i*n+j] = mat[j][i];
}
void reshapeB(int m, int n, double** mat, double* cvec) {
	/*
		Turns a matrix (given by a double pointer) into its C vector format 
		(single vector, rowwise). The matrix "mat" needs to be an n*m matrix
		The vector "vec" is a vector of lenght nm
	*/
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++) 
			mat[j][i]=cvec[i*n+j];
}
void inverse(double** in,double** B, int N)
{
	double* A;
	A=(double*)malloc(sizeof(double)*N*N);
	reshapeF(N,N,in,A);
    int *IPIV = (int*)malloc(sizeof(int)*N+1);
    int LWORK = N*N;
    double *WORK = (double*)malloc(sizeof(double)*LWORK);
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    free(IPIV);
    free(WORK);
    reshapeB(N,N,B,A);
}
void CreshapeF(int m, int n, double complex** mat, double complex* cvec) {
	/*
		Turns a matrix (given by a double pointer) into its C vector format 
		(single vector, rowwise). The matrix "mat" needs to be an n*m matrix
		The vector "vec" is a vector of lenght nm
	*/
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++) 
			cvec[i*n+j] = mat[j][i];
}
void CreshapeB(int m, int n, double complex** mat, double complex* cvec) {
	/*
		Turns a matrix (given by a double pointer) into its C vector format 
		(single vector, rowwise). The matrix "mat" needs to be an n*m matrix
		The vector "vec" is a vector of lenght nm
	*/
	int i,j;
	for (i=0; i<n; i++) 
		for (j=0; j<m; j++) 
			mat[j][i]=cvec[i*n+j];
}
void Cinverse(double complex ** in,double complex ** B, int N)
{
	double complex* A;
	A=(double complex*)malloc(sizeof(double complex)*N*N);
	CreshapeF(N,N,in,A);
    int *IPIV = (int*)malloc(sizeof(int)*N+1);
    int LWORK = N*N;
    double *WORK = (double*)malloc(sizeof(double)*LWORK);
    int INFO;

    zgetrf_(&N,&N,A,&N,IPIV,&INFO);
    zgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

    free(IPIV);
    free(WORK);
    CreshapeB(N,N,B,A);
}