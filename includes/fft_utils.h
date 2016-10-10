#include <fftw3.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "../includes/matrix_operations.h"
void cmat_abs(double complex ** input,int lines,int cols);
int ifft2(double complex** input,double complex** output,int lines,int cols);
int fft2(double complex** input,double complex** output,int lines,int cols);
int rfft2(double ** input,double complex** output,int lines,int cols);
int fftshift(double complex*** input,int lines,int cols);
int ifftshift(double complex*** input,int lines,int cols);
int srfft2(CvMat* input,double complex** output,int lines,int cols);
int srfft2M(CvMat* input,CvMat* input1,double complex** output,int lines,int cols);
int ifft2M(double complex** input,CvMat* output,int lines,int cols);
int ifft2Msp(double complex** input,CvMat* output,CvMat* output1,int lines,int cols);