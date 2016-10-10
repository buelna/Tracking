#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <complex.h>
int get_matrix_order(char * filename, int * c,int * l);
int create_matrix(double ***matrix, int c,int l);
int read_matrix_from_file(char * filename, double *** matrix, int c,int r);
void print_matrix_data(double** matrix,int c,int l);
int write_matrix_into_file(char* file,int** matrix,int c,int l);
int create_cmatrix(double complex***matrix, int c,int l);
void cmatrix_multiplication(double complex **a,double complex **b,double complex** c,int lines,int cols);
void cmatrix_div(double complex **a,double complex **b,double complex** c,int lines,int cols);
void dcmatrix_multiplication(double **a,double complex **b,double complex** c,int lines,int cols);
void matrix_multiplication(double **a,CvMat* b,CvMat* c,int lines,int cols);
void reshape(int m, int n, double** mat, double* cvec);
void creshape(int m, int n, double complex** mat, double complex** cvec,int col);
void inverse(double**,double**,int);
void Cinverse(double complex** in,double complex** B, int N);
void vector_multiplication(int *a,int *b,int* c,int n);
void vsum(int *a,int n,int* sum);
void matXmat(double complex **a,double complex **b,double complex** c,int r1,int c1,int r2,int c2);
void t1matXmat(double complex **a,double complex **b,double complex** c,int r1,int c1,int r2,int c2);
void matXvec(double complex **a,double complex *b,double complex* c,int r1,int c1,int r2,int c2);
void creshapeB(int m, int n, double complex* cvec, double complex** mat);
void matCpy(int m, int n, double complex** in, double complex** out,double mult);
void matsAdd(int m, int n, double complex** a,double complex ** b, double complex** out);
void subsMat(int n, int m, CvMat* in, CvMat* out,double val);
double cMatMean(int n, int m, double complex** in);
void subsMatrix(int n, int m, double complex** in, double complex** out,double val);