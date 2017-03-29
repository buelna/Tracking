#include <math.h>
#include <fftw3.h>
#include <complex.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "../includes/matrix_operations.h"
#define REAL 0 //identifies real part in matrix
#define IMAG 1 //identifies imaginary part in matrix
void cmat_abs(double complex ** input,int lines,int cols)
{
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			//input[i][j]=(creal(input[i][j])>=0)?creal(input[i][j]):(-1*creal(input[i][j]));
			input[i][j]=cabs(input[i][j]);
		}
	}
}
int fft2(double complex ** input,double complex** output,int lines,int cols)
{
	fftw_complex *in,*out;
	fftw_plan plan;
	int i,j,k;
	/*
		Allocate memory for input and output of fourier transform
	*/
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	plan=fftw_plan_dft_2d(lines,cols,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
	/*
		copy the matrix into the structure for dft
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			in[k][REAL]=creal(input[i][j]);
			in[k][IMAG]=cimag(input[i][j]);
			k++;
		}
	}
	/*
		perform the transform
	*/
	fftw_execute(plan);
	/*
		extract the result
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			output[i][j]=out[k][REAL]+out[k][IMAG]*I;
			k++;
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return 1;
}
int rfft2(double ** input,double complex** output,int lines,int cols)
{
	fftw_complex *in,*out;
	fftw_plan plan;
	int i,j,k;
	/*
		Allocate memory for input and output of fourier transform
	*/
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	plan=fftw_plan_dft_2d(lines,cols,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
	/*
		copy the matrix into the structure for dft
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			in[k][REAL]=input[i][j];
			in[k][IMAG]=0*I;
			k++;
		}
	}
	/*
		perform the transform
	*/
	fftw_execute(plan);
	/*
		extract the result
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			output[i][j]=out[k][REAL]+out[k][IMAG]*I;
			k++;
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return 1;
}
int srfft2M(CvMat* input,CvMat* input1,double complex** output,int lines,int cols)
{
	fftw_complex *in,*out;
	fftw_plan plan;
	int i,j,k;
	/*
		Allocate memory for input and output of fourier transform
	*/
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	plan=fftw_plan_dft_2d(lines,cols,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
	/*
		copy the matrix into the structure for dft
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			in[k][REAL]=input->data.fl[i*cols+j];
			in[k][IMAG]=input1->data.fl[i*cols+j]*I;
			k++;
		}
	}
	/*
		perform the transform
	*/
	fftw_execute(plan);
	/*
		extract the result
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			output[i][j]=out[k][REAL]+out[k][IMAG]*I;
			k++;
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return 1;
}
int srfft2(CvMat* input,double complex** output,int lines,int cols)
{
	fftw_complex *in,*out;
	fftw_plan plan;
	int i,j,k;
	/*
		Allocate memory for input and output of fourier transform
	*/
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	plan=fftw_plan_dft_2d(lines,cols,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
	/*
		copy the matrix into the structure for dft
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			in[k][REAL]=input->data.fl[i*cols+j];
			in[k][IMAG]=0*I;
			k++;
		}
	}
	/*
		perform the transform
	*/
	fftw_execute(plan);
	/*
		extract the result
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			output[i][j]=out[k][REAL]+out[k][IMAG]*I;
			k++;
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return 1;
}
int ifft2(double complex** input,double complex** output,int lines,int cols)
{
	fftw_complex *in,*out;
	fftw_plan plan;
	int i,j,k;
	double factor;
	/*
		Allocate memory for input and output of fourier transform
	*/
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	plan=fftw_plan_dft_2d(lines,cols,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
	/*
		copy the matrix into the structure for dft
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			in[k][REAL]=creal(input[i][j]);
			in[k++][IMAG]=cimag(input[i][j]);
		}
	}
	/*
		perform the transform
	*/
	fftw_execute(plan);
	/*
		extract the result
	*/
	k=0;
	factor=lines*cols;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			output[i][j]=(out[k][REAL])+(out[k][IMAG])*I;
			output[i][j]=output[i][j]/factor;
			k++;
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return 1;
}
int ifft2M(double complex** input,CvMat* output,int lines,int cols)
{
	fftw_complex *in,*out;
	fftw_plan plan;
	int i,j,k;
	double factor;
	/*
		Allocate memory for input and output of fourier transform
	*/
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	plan=fftw_plan_dft_2d(lines,cols,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
	/*
		copy the matrix into the structure for dft
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			in[k][REAL]=creal(input[i][j]);
			in[k++][IMAG]=cimag(input[i][j]);
		}
	}
	/*
		perform the transform
	*/
	fftw_execute(plan);
	/*
		extract the result
	*/
	k=0;
	factor=lines*cols;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			output->data.fl[(i*cols)+j]=(out[k][REAL])+(out[k][IMAG])*I;
			output->data.fl[(i*cols)+j]=output->data.fl[(i*cols)+j]/factor;
			k++;
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return 1;
}
int ifft2Msp(double complex** input,CvMat* output,CvMat* output1,int lines,int cols)
{
	fftw_complex *in,*out;
	fftw_plan plan;
	int i,j,k;
	double factor;
	double complex tmp;
	/*
		Allocate memory for input and output of fourier transform
	*/
	in=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	out=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*lines*cols);
	plan=fftw_plan_dft_2d(lines,cols,in,out,FFTW_BACKWARD,FFTW_ESTIMATE);
	/*
		copy the matrix into the structure for dft
	*/
	k=0;
	for (i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			in[k][REAL]=creal(input[i][j]);
			in[k][IMAG]=cimag(input[i][j]);
			k++;
		}
	}
	/*
		perform the transform
	*/
	fftw_execute(plan);
	/*
		extract the result
	*/
	k=0;
	factor=lines*cols;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			tmp=(out[k][REAL])+(out[k][IMAG])*I;
			tmp=tmp/factor;
			output->data.fl[(i*cols)+j]=creal(tmp);
			output1->data.fl[(i*cols)+j]=cimag(tmp);
			k++;
		}
	}
	fftw_destroy_plan(plan);
	fftw_free(in);
	fftw_free(out);
	return 1;
}
int fftshift(double complex*** input,int lines,int cols)
{
	int m2, n2;
	double complex** result;
	create_cmatrix(&result,cols,lines);
	m2 = (lines / 2);    // half of row dimension
	n2 = (cols / 2);    // half of column dimension
	// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
	//assign new quadrant #1
	for (int i = 0; i <m2; ++i)
	{
		for (int j = 0; j <n2; ++j)
		{
			result[i][j]=(*input)[m2+i+1][n2+j+1];
		}
	}
	
	//asssign new quadrant #3
	for (int i = m2; i <lines; ++i)
	{
		for (int j = n2; j <cols; ++j)
		{
			result[i][j]=(*input)[i-m2][j-n2];
		}
	}
	//asssing new quadrant #2
	for (int i = 0; i <m2; ++i)
	{
		for (int j = n2; j <cols; ++j)
		{
			result[i][j]=(*input)[i+m2+1][j-n2];
		}
	}
	//asssing new quadrant #4
	for (int i = m2; i <lines; ++i)
	{
		for (int j = 0; j <n2; ++j)
		{
			result[i][j]=(*input)[i-m2][j+n2+1];
		}
	}
	(*input)=result;
	return 0;
}
int ifftshift(double complex*** input,int lines,int cols)
{
	int m2, n2;
	double complex** result;
	create_cmatrix(&result,cols,lines);
	m2 = (lines / 2);    // half of row dimension
	n2 = (cols / 2);    // half of column dimension
	//assign new quadrant #1
	for (int i = 0; i <=m2; ++i)
	{
		for (int j = 0; j <=n2; ++j)
		{
			result[i][j]=(*input)[m2+i][n2+j];
		}
	}
	//asssign new quadrant #3
	for (int i = m2+1; i <lines; ++i)
	{
		for (int j = n2+1; j <cols; ++j)
		{
			result[i][j]=(*input)[i-m2-1][j-n2-1];
		}
	}
	//asssing new quadrant #2
	for (int i = 0; i <=m2; ++i)
	{
		for (int j = n2+1; j <cols; ++j)
		{
			result[i][j]=(*input)[i+m2][j-n2-1];
		}
	}
	//asssing new quadrant #4
	for (int i = m2+1; i <lines; ++i)
	{
		for (int j = 0; j <=n2; ++j)
		{
			result[i][j]=(*input)[i-m2-1][j+n2];
		}
	}
	(*input)=result;
	return 0;
}
/*
int ifftshift(double complex** input,int lines,int cols)
{
	int m, n;      // FFT row and column dimensions might be different
	m=lines;
	n=cols;
	int m2, n2;
	//int i, k;
	//double complex x[m][n];
	double complex tmp13, tmp24;

	m2 = (m / 2);    // half of row dimension
	n2 = (n / 2);    // half of column dimension
	// interchange entries in 4 quadrants, 1 <--> 3 and 2 <--> 4
	for (int i = 0; i < n; ++i)
	{
		tmp13=input[m2][i];
		for (int j = 0; j <m2; ++j)
		{
			input[m2+j][i]=input[j][i];
			input[j][i]=input[m2+j+1][i];
		}
		input[m-1][i]=tmp13;
	}
	for (int i = 0; i < m; ++i)
	{
		tmp13=input[i][n2];
		for (int j = 0; j < n2; ++j)
		{
			input[i][n2+j]=input[i][j];
			input[i][j]=input[i][n2+j+1];
		}
		input[i][n-1]=tmp13;
	}
	return 0;
}*/