#include <complex.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv/cv.h"
double findMax(double **a,int r, int c,int* x,int* y);
float findMaxMat(CvMat* a,int r, int c);
double complex maxc(double complex **a,int r, int c);
double complex minc(double complex **a,int r, int c);
double max(double **a,int r, int c);
double min(double **a,int r, int c);
void set_zeros(double **cp, int xini,int xend,int yini,int yend);
double calcDCfast(double** cp,int nx,int ny);
void crop_selection(double **cp,double complex ***crop, int xini,int xend,int yini,int yend);
void rcrop_selection(double **cp,double ***crop, int xini,int xend,int yini,int yend);
int read_frame(FILE ** in_file,double **curr_frame);
void gaussian_window(double **w,int sz);
void gaussian_2Dwindow(double **window,int lines,int cols);
void meshgrid(double **matrix,double *wl,double *wc,int lines,int cols);
void cmat2gray(double complex ***matrix,int lines,int cols);
void mat2grayM(CvMat* matrix,int lines,int cols);
void mat2gray(double ***matrix,int lines,int cols);
double mean(double* array,int t);
double stddev(double* array,int t,double mean);
double in_noise_est(CvMat* f,int lines,int cols);
int est_rho(float **img,int lines,int cols,double *rho);
IplImage *rotateImage2(const IplImage *src, float angleDegrees);
IplImage* rotateImage(IplImage* src, float angleDegrees);
void localfalsecrop(double** Imfalse,int sz1,int sz2,double imfx2,double imfy2,int xf1,int yf1,
int fsz1,int fsz2,double**** falsetemp,int* contf,int* szlf1,int* szlf2);
void localcrop(double x1,double y1,int sz1,int sz2,double fimfx2,double fimfy2,double** Im
,double*** scenec,double* cxadp,double* cyadp,int* ablelocal);
void cconj(double complex **in,double complex **out,int lines, int cols);
void prediction(int* ex,int* ey,int numpred,int* tx,int ssqx,int sqqx,int* Pfx1,int* Pfy1);
void normal_crop(int x1,int y1,int Isz1,int Isz2,int imfx2,int imfy2,IplImage* frame,IplImage* prevf,int re2,int co2,
int fsz1,int fsz2,int Pfx1,int Pfy1,IplImage* Imfr,IplImage* crop_scene,int* xadp,int* yadp,int* recc);
double avg(float** array,int t,int cols);
double var(float** array,int t,int cols);



