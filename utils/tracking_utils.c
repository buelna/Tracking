#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "../includes/fft_utils.h"
#include "../includes/matrix_operations.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv/cv.h"
double avg(float** array,int t,int cols)
{
	int i;
	double sum,mean=0.0;
	sum=0;
	for (i = 0; i < t; ++i)
	{
		for(int j=0;j<cols;j++)
		{
			sum+=(*array)[i*cols+j];
		}
	}
	mean=sum/(t*cols);
	return mean;
}
double var(float** array,int t,int cols)
{
	int i,tminus;
	double stddev=0.0;
	double mean=avg(array,t,cols);
	tminus=(t*cols)-1;
	for (i = 0; i < t; ++i)
	{
		for(int j=0;j<cols;j++)
		{
			stddev+=(((*array)[i*cols+j]-mean)*((*array)[i*cols+j]-mean));
		}
	}
	return (stddev/tminus);
}
IplImage *rotateImage2(const IplImage *src, float angleDegrees)
{
    // Create a map_matrix, where the left 2x2 matrix
    // is the transform and the right 2x1 is the dimensions.
    float m[6];
    CvMat M = cvMat(2, 3, CV_32F, m);
    int w = src->width;
    int h = src->height;
    float angleRadians = angleDegrees * ((float)CV_PI / 180.0f);
    m[0] = (float)(cos(angleRadians));
    m[1] = (float)(sin(angleRadians));
    m[3] = -m[1];
    m[4] = m[0];
    m[2] = w*0.5f;
    m[5] = h*0.5f;

    // Make a spare image for the result
    CvSize sizeRotated;
    sizeRotated.width = cvRound(w);
    sizeRotated.height = cvRound(h);

    // Rotate
    IplImage *imageRotated = cvCreateImage(sizeRotated,
        src->depth, src->nChannels);

    // Transform the image
    cvGetQuadrangleSubPix(src, imageRotated, &M);

    return imageRotated;
}
IplImage* rotateImage(IplImage* src, float angle)
{
	CvPoint2D32f srcTri[3],dstTri[3];
	CvMat* rot_mat=cvCreateMat(2,3,CV_32FC1);
	CvMat* warp_mat=cvCreateMat(2,3,CV_32FC1);
	int w,h;
	w=src->width;
	h=src->height;
	IplImage *dst;
	IplImage *imageRotated;
	dst=cvCloneImage(src);
	dst->origin=src->origin;
	cvZero(dst);
	srcTri[0].x=0;
	srcTri[0].y=0;
	srcTri[1].x=w-1;
	srcTri[1].y=0;
	srcTri[2].x=0;
	srcTri[2].y=h-1;
	dstTri[0].x=w*0.0;
	dstTri[0].y=h*0.33;
	dstTri[1].x=w*0.85;
	dstTri[1].y=h*0.25;
	dstTri[2].x=w*0.15;
	dstTri[2].y=h*0.7;
	cvGetAffineTransform(srcTri,dstTri,warp_mat);
    //cvWarpAffine(src, dst,warp_mat,CV_WARP_FILL_OUTLIERS,cvScalarAll(0));
    cvWarpAffine(src, dst,warp_mat,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0));
  	imageRotated=cvCloneImage(src);
    CvPoint2D32f center=cvPoint2D32f(w/2,h/2);
    cv2DRotationMatrix(center,angle,1,rot_mat);
    cvWarpAffine(src,imageRotated,rot_mat,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(0));
    //cvWarpAffine(src,imageRotated,rot_mat,CV_INTER_LINEAR,cvScalarAll(0));
    cvReleaseMat(&rot_mat);
    cvReleaseMat(&warp_mat);
    cvReleaseImage(&dst);
    return imageRotated;
}
double findMax(double **a,int r, int c,int* x,int* y)
{ 
	int i, j;
	double t;
	t = a[0][0];
	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			if(a[i][j] > t){
				t = a[i][j];
				*x=i;
				*y=j;
			}
		}
	}
	return t;
}
float findMaxMat(CvMat* a,int r, int c)
{ 
	int x, y;
	float t;
	t = a->data.fl[0];
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			if(a->data.fl[i*c+j] > t){
				t = a->data.fl[i*c+j];
				x=i;
				y=j;
			}
		}
	}
	printf("%d %d\n",x,y);
	return t;
}
double max(double **a,int r, int c)
{ 
	int i, j;
	double t;
	t = a[0][0];
	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			if((a[i][j]) > (t))
				t = a[i][j];
		}
	}
	return t;
}
float maxM(CvMat* a,int r, int c)
{ 
	int i, j;
	float t;
	t = a->data.fl[0];
	for (i = 0; i < r; i++) {
		for (j = 1; j < c; j++) {
			if((a->data.fl[i*c+j]) > (t))
				t = a->data.fl[i*c+j];
		}
	}
	return t;
}
float minM(CvMat* a,int r, int c)
{ 
	int i, j;
	float t;
	t = a->data.fl[0];
	for (i = 0; i < r; i++) {
		for (j = 1; j < c; j++) {
			if((a->data.fl[i*c+j]) < (t))
				t = a->data.fl[i*c+j];
		}
	}
	return t;
}
double min(double **a,int r, int c)
{ 
	int i, j;
	double t;
	t = a[0][0];
	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			if((a[i][j]) < (t))
				t = a[i][j];
		}
	}
	return t;
}
double complex maxc(double complex **a,int r, int c)
{ 
	int i, j;
	double complex t;
	t = a[0][0];
	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			if(abs(creal(a[i][j])) > abs(creal(t)))
				t = a[i][j];
		}
	}
	return t;
}
double complex minc(double complex **a,int r, int c)
{ 
	int i, j;
	double complex t;
	t = a[0][0];
	for (i = 0; i < r; i++) {
		for (j = 0; j < c; j++) {
			if(abs(creal(a[i][j])) < abs(creal(t)))
				t = a[i][j];
		}
	}
	return t;
}
void set_zeros(double **cp, int xini,int xend,int yini,int yend)
{
	int i,j;
	for (i = xini; i <=xend; ++i)
	{
		for (j = yini; j <= yend; ++j)
		{
			cp[i][j]=0;
		}
	}
}
double calcDCfast(double** cp,int nx,int ny)
{
	double cpT,cpB,DC,**cpc;
	int x0,y0,k=20;
	int lxl,lxr,lyl,lyr;
	create_matrix(&cpc,ny,nx);
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j)
		{
			cpc[i][j]=cp[i][j];
		}
	}
	cpT=findMax(cpc,nx,ny,&x0,&y0);
	lxl=x0-k;
	lxr=x0+k;
	lyl=y0-k;
	lyr=y0+k;
	if ((lxl>0 && lxr<nx)&&(lyl>0 && lyr < ny))
	{
		set_zeros(cpc,lxl,lxr,lyl,lyr);
		cpB=max(cpc,nx,ny);
		DC=1-(cpB/cpT);
	}
	else
		DC=0;
	free(cpc);
	return DC;
}
void crop_selection(double **cp,double complex ***crop, int xini,int xend,int yini,int yend)
{
	int i,j,szx,szy;
	xini--;xend--;
	szx=xend-xini+1;
	szy=yend-yini+1;
	for (i = 0; i <=szx; ++i)
	{
		for (j = 0; j <= szy; ++j)
		{
			(*crop)[i][j]=cp[i+xini][j+yini]+0*I;
		}
	}
}
void rcrop_selection(double **cp,double ***crop, int xini,int xend,int yini,int yend)
{
	int i,j,szx,szy;
	xini--;xend--;
	szx=xend-xini+1;
	szy=yend-yini+1;
	for (i = 0; i <=szx; ++i)
	{
		for (j = 0; j <= szy; ++j)
		{
			(*crop)[i][j]=cp[i+xini][j+yini]+0*I;
		}
	}
}
int read_frame(FILE ** in_file,double **curr_frame){
	double tmp;
	int i,j;
	for (i = 0; i < 480; ++i)
	{
		for (j= 0; j < 640; ++j)
		{
			fscanf(*in_file,"%lf",&tmp);
			curr_frame[i][j]=tmp/255;
		}
	}
	return 1;
}
void gaussian_window(double **w,int sz){
	int order,n2,i;
	double a,n,tmp,sq;
	a=2.5;
	order=sz;
	n2=sz/2;
	//(*w)=(double *)malloc(sz*sizeof(double));
	for (i = 0; i < order; ++i)
	{
		n=i-n2;
		tmp=((a*n)/n2);
		sq=tmp*tmp;
		(*w)[i]=exp(-0.5*sq);
	}
}
void meshgrid(double **matrix,double *wl,double *wc,int lines,int cols)
{
	int i,j;
	for (i = 0; i < lines; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			matrix[i][j]=wc[i]*wl[j];
		}
	}
}
void gaussian_2Dwindow(double **window,int lines,int cols)
{
	double *wl,*wc;
	wl=(double *)malloc(cols*sizeof(double));
	wc=(double *)malloc(lines*sizeof(double));
	gaussian_window(&wl,cols);
	gaussian_window(&wc,lines);
	meshgrid(window,wl,wc,lines,cols);
}
void mat2gray(double ***matrix,int lines,int cols){
	int i,j;
	double tmp,minv,maxv;
	minv=min(*matrix,lines,cols);
	maxv=max(*matrix,lines,cols);

	for (i = 0; i <lines ; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			tmp=(*matrix)[i][j];
			(*matrix)[i][j]=(tmp-minv)/(maxv-minv);
		}
	}
}
void mat2grayM(CvMat* matrix,int lines,int cols){
	int i,j;
	float tmp,minv,maxv;
	minv=minM(matrix,lines,cols);
	maxv=maxM(matrix,lines,cols);
	for (i = 0; i <lines ; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			tmp=matrix->data.fl[i*cols+j];
			matrix->data.fl[i*cols+j]=(tmp-minv)/(maxv-minv);
		}
	}
}
void cmat2gray(double complex ***matrix,int lines,int cols){
	int i,j;
	double complex tmp,minv,maxv;
	minv=minc(*matrix,lines,cols);
	maxv=maxc(*matrix,lines,cols);

	for (i = 0; i <lines ; ++i)
	{
		for (j = 0; j < cols; ++j)
		{
			tmp=(*matrix)[i][j];
			(*matrix)[i][j]=(tmp-minv)/(maxv-minv);
		}
	}
}
double mean(double* array,int t)
{
	int i;
	double sum,mean=0.0;
	sum=0;
	for (i = 0; i < t; ++i)
	{
		sum+=array[i];
	}
	mean=sum/t;
	return mean;
}
double stddev(double* array,int t,double mean)
{
	int i,tminus;
	double stddev=0.0;
	tminus=t-1;
	for (i = 0; i < t; ++i)
	{
		stddev+=((array[i]-mean)*(array[i]-mean));
	}
	return sqrt(stddev/tminus);
}
void cconj(double complex **in,double complex **out,int lines, int cols)
{
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			out[i][j]=~in[i][j];
		}
	}
}
double in_noise_est(CvMat* f,int lines,int cols)
{
	double complex **F,**Fconj,**ff;
	double maxv,Rs0,sig;
	create_cmatrix(&Fconj, cols,lines);
	create_cmatrix(&F, cols,lines);
	create_cmatrix(&ff, cols,lines);
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			ff[i][j]=(double)f->data.fl[i*cols+j];
		}
	}
	fft2(ff,F,lines,cols);
	cconj(F,Fconj,lines,cols);
	cmatrix_multiplication(F,Fconj,Fconj,lines,cols);
	ifft2(Fconj,F,lines,cols);
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			F[i][j]=cabs(F[i][j]);
			/*if (creal(F[i][j])<0)
			{
				F[i][j]=-1*creal(F[i][j]);
			}*/
		}
	}
	maxv=creal(maxc(F,lines,cols));
	for (int i = 0; i < lines; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			F[i][j]=F[i][j]/maxv;
		}
	}
	Rs0=2*creal(F[0][1])-creal(F[0][2]);
	sig=F[0][0]-Rs0;
	sig=cabs(sig);
	/*if (sig<0)
	{
		sig=-1*sig;
	}*/
	free(Fconj);
	free(ff);
	free(F);
	return sig;
}
int est_rho(float **img,int lines,int cols,double *rho)
{
	int n,line,col;
	double *x,*y,mx,my,stdx,stdy,p,*stdX,*stdY;
	line=floor(lines/2)-1;
	col=floor(cols/2)-1;
	/*
		allocate memory for x and y
	*/
	x=(double *)malloc(sizeof(double)*cols);
	y=(double *)malloc(sizeof(double)*lines);
	if (!(x))
	{
		return 0;
	}
	if (!(y))
	{
		return 0;
	}
	/*
		copy line and column to x and y
	*/
	for (int i = 0; i < cols; ++i)
	{
		x[i]=(*img)[line*cols+i];
	}
	for (int i = 0; i < lines; ++i)
	{
		y[i]=(*img)[i*cols+col];
	}
	/*
		calculate mean of x and y
	*/
	mx=mean(x,cols);
	my=mean(y,lines);
	/*
		estimate standard deviation
	*/
	stdx=stddev(x,cols,mx);
	stdy=stddev(y,lines,my);

	if (lines<=cols)
	{
		n=lines;
	}
	else
	{
		n=cols;
	}
	/*
		allocate memory for stdX and stdY
	*/
	stdX=(double *)malloc(sizeof(double)*n);
	stdY=(double *)malloc(sizeof(double)*n);
	if (!(stdX))
	{
		return 0;
	}
	if (!(stdY))
	{
		return 0;
	}
	/*
		determine p
	*/
	p=0.0;
	for (int i = 0; i < n; ++i)
	{
		stdX[i]=(x[i]-mx)/stdx;
		stdY[i]=(y[i]-my)/stdy;
		p=p+(stdX[i]*stdY[i]);
	}
	if (p<0)
	{
		p=-1*p;
	}
	*rho=p/(n-1);
	return 1;
}
void localcrop(double x1,double y1,int sz1,int sz2,double fimfx2,double fimfy2,double** Im
,double*** scenec,double* cxadp,double* cyadp,int* ablelocal)
{
	double xres,yres;
	int l,c;
	if ((fimfx2*2)<sz1 && (fimfy2*2)<sz2)
	{
		*ablelocal=1;

		//normal case
		if ((x1-fimfx2)>0 && (x1+fimfx2-1) < sz1&& (y1-fimfy2)>0 && (y1+fimfy2-1) < sz2)
		{
			l=abs((x1+fimfx2-1)-(x1-fimfx2))+1;
			c=abs((y1+fimfy2-1)-(y1-fimfy2))+1;
			create_matrix(scenec,c,l);
			rcrop_selection(Im,scenec, x1-fimfx2,x1+fimfx2-1,y1-fimfy2,y1+fimfy2-1);
			*cxadp=0;
			*cyadp=0;
		}
		else
		{
			//case when it overflows the upper side(x1-fimfx2)<0
			if ((x1-fimfx2)<=0 && (x1+fimfx2-1) < sz1 && (y1-fimfy2)>0 && (y1+fimfy2-1) < sz2)
			{
				xres=fimfx2;
				l=xres+fimfx2;
				c=abs((y1+fimfy2-1)-(y1-fimfy2))+1;	
				create_matrix(scenec,c,l);
				rcrop_selection(Im,scenec,0,xres+fimfx2,y1-fimfy2,y1+fimfy2-1);
				*cxadp=floor(fimfx2)-x1+2;
				*cyadp=0;
			}
			else
			{
				//case when it overflows the lower side (x1+fimfx2-1)>480
				if ((x1-fimfx2)>0 && (x1+fimfx2-1)>=sz1 && (y1-fimfy2)>0 && (y1+fimfy2-1) < sz2)
				{
					xres=sz1-fimfx2;
					l=abs((xres+fimfx2-1)-(xres-fimfx2))+1;
					c=abs((y1+fimfy2-1)-(y1-fimfy2))+1;
					create_matrix(scenec,c,l);
					rcrop_selection(Im,scenec,xres-fimfx2,xres+fimfx2-1,y1-fimfy2,y1+fimfy2-1);
					*cxadp= -x1+2 + (floor(fimfx2)) + sz1 - 2*fimfx2;
					*cyadp=0;
				}
				else
				{
					//when it overflows at the left side(y1-fimfy2)<0
					if ((x1-fimfx2)>0 && (x1+fimfx2-1)<sz1&& (y1-fimfy2)<=0 && (y1+fimfy2-1) < sz2)
					{
						yres=fimfy2;
						l=abs((x1+fimfx2-1)-(x1-fimfx2))+1;
						c=yres+fimfy2;
						create_matrix(scenec,c,l);
						rcrop_selection(Im,scenec,x1-fimfx2,x1+fimfx2-1,0,yres+fimfy2);
						*cxadp=0;
						*cyadp=floor(fimfy2)-y1+2;
					}
					else
					{
						//when it overflows at the right side(y1+fimfy2-1)> (640+sv*2)
						if ((x1-fimfx2)>0 && (x1+fimfx2-1)<sz1 && (y1-fimfy2)>0 && (y1+fimfy2-1)>= sz2)
						{
							yres=sz2-fimfy2;
							l=abs((x1+fimfx2-1)-(x1-fimfx2))+1;
							c=abs((yres+fimfy2-1)-(yres-fimfy2))+1;
							create_matrix(scenec,c,l);
							rcrop_selection(Im,scenec,x1-fimfx2,x1+fimfx2-1,yres-fimfy2,yres+fimfy2-1);
							*cyadp= -y1+2 + (floor(fimfy2)) + sz2 - 2*fimfy2;
							*cxadp=0;
						}
						else
						{
							//upper left corner
							if ((x1-fimfx2)<=0 && (x1+fimfx2-1)<sz1 && (y1-fimfy2)<=0 && (y1+fimfy2-1)< sz2)
							{
								xres=fimfx2;
								yres=fimfy2;
								l=xres+fimfx2;
								c=yres+fimfy2;
								create_matrix(scenec,c,l);
								rcrop_selection(Im,scenec,0,xres+fimfx2,0,yres+fimfy2);
								*cyadp=floor(fimfy2)-y1+2;
								*cxadp=floor(fimfx2)-x1+2;
							}
							else
							{
								//Upper right corner
								if ((x1-fimfx2)<=0 && (x1+fimfx2-1)<sz1 && (y1-fimfy2)>0 && (y1+fimfy2-1)>= sz2)
								{
									xres=fimfx2;
									yres=sz2-fimfy2;
									l=xres+fimfx2;
									c=abs((yres+fimfy2-1)-(yres-fimfy2))+1;
									create_matrix(scenec,c,l);
									rcrop_selection(Im,scenec,0,xres+fimfx2,yres-fimfy2,yres+fimfy2-1);
									*cyadp= -y1+2 + (floor(fimfy2)) + sz2 - 2*fimfy2;
									*cxadp=floor(fimfx2)-x1+2;
								}
								else
								{
									//Lower left corner
									if ((x1-fimfx2)>0 && (x1+fimfx2-1)>=sz1&& (y1-fimfy2)<=0 && (y1+fimfy2-1)< sz2)
									{
										xres=sz1-fimfx2;
										yres=fimfy2;
										l=abs((xres+fimfx2-1)-(xres-fimfx2))+1;
										c=yres+fimfy2;
										create_matrix(scenec,c,l);
										rcrop_selection(Im,scenec,xres-fimfx2,xres+fimfx2-1,0,yres+fimfy2);
										*cyadp=floor(fimfy2)-y1+2;
										*cxadp=-x1+2 + (floor(fimfx2)) + sz1 - 2*fimfx2;
									}
									else
									{
										//Lower right corner
										if ((x1-fimfx2)>0 && (x1+fimfx2-1)>=sz1 && (y1-fimfy2)>0 && (y1+fimfy2-1)>= sz2)
										xres=sz1-fimfx2;
										yres=sz2-fimfy2;
										l=abs((xres+fimfx2-1)-(xres-fimfx2))+1;
										c=abs((yres+fimfy2-1)-(yres-fimfy2))+1;
										create_matrix(scenec,c,l);
										rcrop_selection(Im,scenec,xres-fimfx2,xres+fimfx2-1,yres-fimfy2,yres+fimfy2-1);
										*cyadp=-y1+2 + (floor(fimfy2)) + sz2 - 2*fimfy2;
										*cxadp=-x1+2 + (floor(fimfx2)) + sz1 - 2*fimfx2;
									}
								}
							}
						}
					}
				}
			}
		}

	}
	else
	{
		*ablelocal=0;
		*cyadp=0;
		*cxadp=0;
	}
}
void localfalsecrop(double** Imfalse,int sz1,int sz2,double imfx2,double imfy2,int xf1,int yf1,
int fsz1,int fsz2,double**** falsetemp,int* contf,int* szlf1,int* szlf2)
{
	double*** tmpMat;
	fsz1=fsz1-5;
	fsz2=fsz2-5;
	*contf=0;
	*szlf1=0;//lines
	*szlf2=0;//cols
    //Upper side crop (x1-imfx2)<0
    if ((xf1-imfx2-fsz1)>0 && (xf1+imfx2-1-fsz1) < sz1 && (yf1-imfy2)>0 && (yf1+imfy2-1) < sz2)
    {   
		*szlf1=abs((xf1-imfx2-fsz1)-(xf1+imfx2-1-fsz1))+1;
		*szlf2=abs((yf1-imfy2)-(yf1+imfy2-1))+1;
		create_matrix(tmpMat,*szlf2,*szlf1);
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1,xf1+imfx2-1-fsz1,yf1-imfy2,yf1+imfy2-1);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	//Lower side crop (x1+imfx2-1)>480
	if ((xf1-imfx2+fsz1)>0 && (xf1+imfx2-1+fsz1) < sz1 && (yf1-imfy2)>0 && (yf1+imfy2-1) < sz2)
	{
		if (*szlf1==0 || *szlf2==0)
		{
			*szlf1=abs((xf1-imfx2+fsz1)-(xf1+imfx2-1+fsz1))+1;
			*szlf2=abs((yf1-imfy2)-(yf1+imfy2-1))+1;
			create_matrix(tmpMat,*szlf2,*szlf1);
		}
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1,xf1+imfx2-1+fsz1,yf1-imfy2,yf1+imfy2-1);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	//Left side crop(y1-imfy2)<0
	if ((xf1-imfx2)>0 && (xf1+imfx2-1) < sz1 && (yf1-imfy2-fsz2)>0 && (yf1+imfy2-1-fsz2) < sz2)
	{
		if (*szlf1==0 || *szlf2==0)
		{
			*szlf1=abs((xf1-imfx2)-(xf1+imfx2-1))+1;
			*szlf2=abs((yf1-imfy2-fsz2)-(yf1+imfy2-1-fsz2))+1;
			create_matrix(tmpMat,*szlf2,*szlf1);
		}
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2,xf1+imfx2-1,yf1-imfy2-fsz2,yf1+imfy2-1-fsz2);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	//right side crop (y1+imfy2-1)> (640+sv*2)
	if ((xf1-imfx2)>0 && (xf1+imfx2-1) < sz1 && (yf1-imfy2+fsz2)>0 && (yf1+imfy2-1+fsz2) < sz2)
	{
		if (*szlf1==0 || *szlf2==0)
		{
			*szlf1=abs((xf1-imfx2)-(xf1+imfx2-1))+1;
			*szlf2=abs((yf1-imfy2+fsz2)-(yf1+imfy2-1+fsz2))+1;
			create_matrix(tmpMat,*szlf2,*szlf1);
		}
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2,xf1+imfx2-1,yf1-imfy2+fsz2,yf1+imfy2-1+fsz2);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	//Upper left corner crop
	if ((xf1-imfx2-fsz1)>0 && (xf1+imfx2-1-fsz1) < sz1 && (yf1-imfy2-fsz2)>0 && (yf1+imfy2-1-fsz2) < sz2)
	{
		if (*szlf1==0 || *szlf2==0)
		{
			*szlf1=abs((xf1-imfx2-fsz1)-(xf1+imfx2-1-fsz1))+1;
			*szlf2=abs((yf1-imfy2-fsz2)-(yf1+imfy2-1-fsz2))+1;
			create_matrix(tmpMat,*szlf2,*szlf1);
		}
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1,xf1+imfx2-1-fsz1,yf1-imfy2-fsz2,yf1+imfy2-1-fsz2);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	//Upper right corner crop
	if ((xf1-imfx2-fsz1)>0 && (xf1+imfx2-1-fsz1) < sz1 && (yf1-imfy2+fsz2)>0 && (yf1+imfy2-1+fsz2) < sz2)
	{
		if (*szlf1==0 || *szlf2==0)
		{
			*szlf1=abs((xf1-imfx2-fsz1)-(xf1+imfx2-1-fsz1))+1;
			*szlf2=abs((yf1-imfy2+fsz2)-(yf1+imfy2-1+fsz2))+1;
			create_matrix(tmpMat,*szlf2,*szlf1);
		}
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1,xf1+imfx2-1-fsz1,yf1-imfy2+fsz2,yf1+imfy2-1+fsz2);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	//lower left corner crop
	if ((xf1-imfx2+fsz1)>0 && (xf1+imfx2-1+fsz1) < sz1 && (yf1-imfy2-fsz2)>0 && (yf1+imfy2-1-fsz2) < sz2)
	{
		if (*szlf1==0 || *szlf2==0)
		{
			*szlf1=abs((xf1-imfx2+fsz1)-(xf1+imfx2-1+fsz1))+1;
			*szlf2=abs((yf1-imfy2-fsz2)-(yf1+imfy2-1-fsz2))+1;
			create_matrix(tmpMat,*szlf2,*szlf1);
		}
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1,xf1+imfx2-1+fsz1,yf1-imfy2-fsz2,yf1+imfy2-1-fsz2);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	//Lower right corner crop
	if ((xf1-imfx2+fsz1)>0 && (xf1+imfx2-1+fsz1) < sz1 && (yf1-imfy2+fsz2)>0 && (yf1+imfy2-1+fsz2) < sz2)
	{
		if (*szlf1==0 || *szlf2==0)
		{
			*szlf1=abs((xf1-imfx2+fsz1)-(xf1+imfx2-1+fsz1))+1;
			*szlf2=abs((yf1-imfy2+fsz2)-(yf1+imfy2-1+fsz2))+1;
			create_matrix(tmpMat,*szlf2,*szlf1);
		}
		rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1,xf1+imfx2-1+fsz1,yf1-imfy2+fsz2,yf1+imfy2-1+fsz2);
		falsetemp[*contf]=tmpMat;
		*contf=*contf+1;
	}

	// Object overflows on the upper side
	if ((xf1-imfx2-fsz1)<=0)
	{
		//lower crop
		if ((xf1-imfx2+fsz1+imfx2)>0 && (xf1+imfx2-1+fsz1+imfx2) < sz1 && (yf1-imfy2)>0 && (yf1+imfy2-1) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2+fsz1+imfx2)-(xf1+imfx2-1+fsz1+imfx2))+1;
				*szlf2=abs((yf1-imfy2)-(yf1+imfy2-1))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1+imfx2,xf1+imfx2-1+fsz1+imfx2,yf1-imfy2,yf1+imfy2-1);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
		//Lower left corner crop
		if ((xf1-imfx2+fsz1+imfx2)>0 && (xf1+imfx2-1+fsz1+imfx2) < sz1 && (yf1-imfy2-fsz2)>0 && (yf1+imfy2-1-fsz2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2+fsz1+imfx2)-(xf1+imfx2-1+fsz1+imfx2))+1;
				*szlf2=abs((yf1-imfy2-fsz2)-(yf1+imfy2-1-fsz2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1+imfx2,xf1+imfx2-1+fsz1+imfx2,yf1-imfy2-fsz2,yf1+imfy2-1-fsz2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
		//lower right corner crop
		if ((xf1-imfx2+fsz1+imfx2)>0 && (xf1+imfx2-1+fsz1+imfx2) < sz1 && (yf1-imfy2+fsz2)>0 && (yf1+imfy2-1+fsz2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2+fsz1+imfx2)-(xf1+imfx2-1+fsz1+imfx2))+1;
				*szlf2=abs((yf1-imfy2+fsz2)-(yf1+imfy2-1+fsz2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1+imfx2,xf1+imfx2-1+fsz1+imfx2,yf1-imfy2+fsz2,yf1+imfy2-1+fsz2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
	}

	// Object overflows on the lower side
	if ((xf1+imfx2-1+fsz1) >= sz1)
	{ 
		//upper side crop (x1-imfx2)<0
		if ((xf1-imfx2-fsz1-imfx2)>0 && (xf1+imfx2-1-fsz1-imfx2) < sz1 && (yf1-imfy2)>0 && (yf1+imfy2-1) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2-fsz1-imfx2)-(xf1+imfx2-1-fsz1-imfx2))+1;
				*szlf2=abs(( yf1-imfy2)-(yf1+imfy2-1))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1-imfx2,xf1+imfx2-1-fsz1-imfx2,yf1-imfy2,yf1+imfy2-1);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}

		//upper left corner crop
		if ((xf1-imfx2-fsz1-imfx2)>0 && (xf1+imfx2-1-fsz1-imfx2) < sz1 && (yf1-imfy2-fsz2)>0 && (yf1+imfy2-1-fsz2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2-fsz1-imfx2)-(xf1+imfx2-1-fsz1-imfx2))+1;
				*szlf2=abs((yf1-imfy2-fsz2)-(yf1+imfy2-1-fsz2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1-imfx2,xf1+imfx2-1-fsz1-imfx2,yf1-imfy2-fsz2,yf1+imfy2-1-fsz2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}

		//upper right corner crop
		if ((xf1-imfx2-fsz1-imfx2)>0 && (xf1+imfx2-1-fsz1-imfx2) < sz1 && (yf1-imfy2+fsz2)>0 && (yf1+imfy2-1+fsz2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2-fsz1-imfx2)-(xf1+imfx2-1-fsz1-imfx2))+1;
				*szlf2=abs((yf1-imfy2+fsz2)-(yf1+imfy2-1+fsz2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1-imfx2,xf1+imfx2-1-fsz1-imfx2,yf1-imfy2+fsz2,yf1+imfy2-1+fsz2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
	}

	//object overflows on the left side
	if ((yf1-imfy2+fsz2)<=0)
	{
		//right side crop(y1+imfy2-1)> (640+sv*2)
		if ((xf1-imfx2)>0 && (xf1+imfx2-1) < sz1&& (yf1-imfy2+fsz2+imfy2)>0 && (yf1+imfy2-1+fsz2+imfy2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2)-(xf1+imfx2-1))+1;
				*szlf2=abs((yf1-imfy2+fsz2+imfy2)-(yf1+imfy2-1+fsz2+imfy2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2,xf1+imfx2-1,yf1-imfy2+fsz2+imfy2,yf1+imfy2-1+fsz2+imfy2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
		//upper right corner crop
		if ((xf1-imfx2-fsz1)>0 && (xf1+imfx2-1-fsz1) < sz1 && (yf1-imfy2+fsz2+imfy2)>0 && (yf1+imfy2-1+fsz2+imfy2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2-fsz1)-(xf1+imfx2-1-fsz1))+1;
				*szlf2=abs((yf1-imfy2+fsz2+imfy2)-(yf1+imfy2-1+fsz2+imfy2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1,xf1+imfx2-1-fsz1,yf1-imfy2+fsz2+imfy2,yf1+imfy2-1+fsz2+imfy2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
		//lower right corner crop
		if ((xf1-imfx2+fsz1)>0 && (xf1+imfx2-1+fsz1) < sz1 && (yf1-imfy2+fsz2+imfy2)>0 && (yf1+imfy2-1+fsz2+imfy2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2+fsz1)-(xf1+imfx2-1+fsz1))+1;
				*szlf2=abs((yf1-imfy2+fsz2+imfy2)-(yf1+imfy2-1+fsz2+imfy2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1,xf1+imfx2-1+fsz1,yf1-imfy2+fsz2+imfy2,yf1+imfy2-1+fsz2+imfy2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
	}

	//object overflows on the right side

	if ((yf1+imfy2-1+fsz2) > sz2)
	{
		//left side crop(y1-imfy2)<0
		if ((xf1-imfx2)>0 && (xf1+imfx2-1) < sz1 && (yf1-imfy2-fsz2-imfy2)>0 && (yf1+imfy2-1-fsz2-imfy2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2)-(xf1+imfx2-1))+1;
				*szlf2=abs((yf1-imfy2-fsz2-imfy2)-(yf1+imfy2-1-fsz2-imfy2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2,xf1+imfx2-1, yf1-imfy2-fsz2-imfy2,yf1+imfy2-1-fsz2-imfy2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
		//upper left corner crop
		if ((xf1-imfx2-fsz1)>0 && (xf1+imfx2-1-fsz1) < sz1 && (yf1-imfy2-fsz2-imfy2)>0 && (yf1+imfy2-1-fsz2-imfy2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2-fsz1)-(xf1+imfx2-1-fsz1))+1;
				*szlf2=abs((yf1-imfy2-fsz2-imfy2)-(yf1+imfy2-1-fsz2-imfy2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2-fsz1,xf1+imfx2-1-fsz1,yf1-imfy2-fsz2-imfy2,yf1+imfy2-1-fsz2-imfy2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
		//lower left corner crop
		if ((xf1-imfx2+fsz1)>0 && (xf1+imfx2-1+fsz1) < sz1 && (yf1-imfy2-fsz2-imfy2)>0 && (yf1+imfy2-1-fsz2-imfy2) < sz2)
		{
			if (*szlf1==0 || *szlf2==0)
			{
				*szlf1=abs((xf1-imfx2+fsz1)-(xf1+imfx2-1+fsz1))+1;
				*szlf2=abs((yf1-imfy2-fsz2-imfy2)-(yf1+imfy2-1-fsz2-imfy2))+1;
				create_matrix(tmpMat,*szlf2,*szlf1);
			}
			rcrop_selection(Imfalse,tmpMat,xf1-imfx2+fsz1,xf1+imfx2-1+fsz1,yf1-imfy2-fsz2-imfy2,yf1+imfy2-1-fsz2-imfy2);
			falsetemp[*contf]=tmpMat;
			*contf=*contf+1;
		}
	}
}
void prediction(int* ex,int* ey,int numpred,int* tx,int ssqx,int sqqx,int* Pfx1,int* Pfy1)
{
	int mtxex[5]={0,0,0,0,0},mtxey[5]={0,0,0,0,0};
	int smtxex,smtxey;
	int mqtxex[5]={0,0,0,0,0},mqtxey[5]={0,0,0,0,0},tx2[5]={4,1,0,1,4},tx4[5]={16,1,0,1,16};
	int smqtxex,smqtxey;
	int smex,smey;
	double cx,ax,cy,ay,bx,by;
	vector_multiplication(ex,tx,mtxex,numpred);
	vector_multiplication(ey,tx,mtxey,numpred);
	vsum(mtxex,numpred,&smtxex);
	vsum(mtxey,numpred,&smtxey);
	vector_multiplication(ex,tx2,mqtxex,numpred);
	vector_multiplication(ey,tx2,mqtxey,numpred);
	vsum(mqtxex,numpred,&smqtxex);
	vsum(mqtxey,numpred,&smqtxey);
	vsum(ex,numpred,&smex);
	vsum(ey,numpred,&smey);

	cx=(numpred*smqtxex-smex*ssqx)/(numpred*sqqx-ssqx*ssqx);
	ax=(smex-cx*ssqx)/numpred;
	cy=(numpred*smqtxey-smey*ssqx)/(numpred*sqqx-ssqx*ssqx);
	ay=(smey-cy*ssqx)/numpred;
	bx=smtxex/ssqx;
	by=smtxey/ssqx;

	*Pfx1=ax+(bx*3)+(cx*9);
	*Pfy1=ay+(by*3)+(cy*9);
}
void normal_crop(int x1,int y1,int Isz1,int Isz2,int imfx2,int imfy2,IplImage* frame,IplImage* prevf,int re2,int co2,
int fsz1,int fsz2,int Pfx1,int Pfy1,IplImage* Imfr,IplImage* crop_scene,int* xadp,int* yadp,int* recc)
{
	//Crop Scene
	//normal case
	int xres,yres,sz1,sz2,cresx,cresy,tempo;
	tempo=x1;
	x1=y1;
	y1=tempo;
	tempo=Pfx1;
	Pfx1=Pfy1;
	Pfy1=tempo;
	CvMat* tempMat;
	IplImage* tmp;
	if ((Pfx1-imfx2)>=0 && (Pfx1+imfx2-1) < Isz1 && (Pfy1-imfy2)>=0 && (Pfy1+imfy2-1) < Isz2)
	{
		IplImage* img2;
		*xadp=0;
		*yadp=0;
		cvSetImageROI(frame, cvRect(Pfx1-imfx2,Pfy1-imfy2,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
		//rcrop_selection(frame,&crop_scene,Pfx1-imfx2,Pfx1+imfx2-1,Pfy1-imfy2,Pfy1+imfy2-1);
	}

	//case when it overflows on the upper side (Pfx1-imfx2)<0
	if ((Pfx1-imfx2)<=0 && (Pfx1+imfx2-1) < Isz1 && (Pfy1-imfy2)>0 && (Pfy1+imfy2-1) < Isz2)
	{
		IplImage* img2;
		xres=imfx2+1;
		*xadp=floor(imfx2)-Pfx1+1;
		*yadp=0;
		cvSetImageROI(frame, cvRect(xres-imfx2,Pfy1-imfy2,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
		//rcrop_selection(frame,&crop_scene,xres-imfx2,xres+imfx2-1,Pfy1-imfy2,Pfy1+imfy2-1);
	}

	//case when it overflows on the lower side (Pfx1+imfx2-1)>480
	if ((Pfx1-imfx2)>0 && (Pfx1+imfx2-1)>=Isz1 && (Pfy1-imfy2)>0 && (Pfy1+imfy2-1) < Isz2)
	{	
		IplImage* img2;
		xres=Isz1-imfx2;
		*xadp=(Pfx1-xres)*(-1);
		*yadp=0;
		cvSetImageROI(frame, cvRect(xres-imfx2,Pfy1-imfy2,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
//		rcrop_selection(frame,&crop_scene,xres-imfx2,xres+imfx2-1,Pfy1-imfy2,Pfy1+imfy2-1);
	}

	//case when it overflows on the left side(Pfy1-imfy2)<0
	if ((Pfx1-imfx2)>0 && (Pfx1+imfx2-1)<Isz1 && (Pfy1-imfy2)<=0 && (Pfy1+imfy2-1) < Isz2)
	{
		IplImage* img2;
		yres=imfy2+1;
		*xadp=0;
		*yadp=floor(imfy2)-Pfy1+1;
		cvSetImageROI(frame, cvRect(Pfx1-imfx2,yres-imfy2,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
//		rcrop_selection(frame,&crop_scene,Pfx1-imfx2,Pfx1+imfx2-1,yres-imfy2,yres+imfy2-1);
	}

	//Overfow on the right side (Pfy1+imfy2-1)> (640+sv*2)
	if ((Pfx1-imfx2)>0 && (Pfx1+imfx2-1)<Isz1&& (Pfy1-imfy2)>0 && (Pfy1+imfy2-1)>= Isz2)
	{
		IplImage* img2;
		yres=Isz2-imfy2;
		*yadp=(Pfy1-yres)*(-1);
		*xadp=0;
		cvSetImageROI(frame, cvRect(Pfx1-imfx2,yres-imfy2,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
//		rcrop_selection(frame,&crop_scene,Pfx1-imfx2,Pfx1+imfx2-1,yres-imfy2,yres+imfy2-1);
	}

	//upper left corner
	if ((Pfx1-imfx2)<=0 && (Pfx1+imfx2-1)<Isz1 && (Pfy1-imfy2)<=0 && (Pfy1+imfy2-1)< Isz2)
	{	
		IplImage* img2;
		xres=imfx2;
		yres=imfy2;
		*xadp=floor(imfx2)-Pfx1+1;
		*yadp=floor(imfy2)-Pfy1+1;
		cvSetImageROI(frame, cvRect(0,0,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
//		rcrop_selection(frame,&crop_scene,0,xres+imfx2-1,0,yres+imfy2-1);
	}

	//Upper right corner
	if ((Pfx1-imfx2)<=0 && (Pfx1+imfx2-1)<Isz1 && (Pfy1-imfy2)>0 && (Pfy1+imfy2-1)>= Isz2)
	{	
		IplImage* img2;
		xres=imfx2;
		yres=Isz2-imfy2;
		*xadp=floor(imfx2)-Pfx1+1;
		*yadp=(Pfy1-yres)*(-1);
		cvSetImageROI(frame, cvRect(0,yres-imfy2,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
//		rcrop_selection(frame,&crop_scene,0,xres+imfx2,yres-imfy2,yres+imfy2-1);
	}

	//Lower left corner
	if ((Pfx1-imfx2)>0 && (Pfx1+imfx2-1)>=Isz1 && (Pfy1-imfy2)<=0 && (Pfy1+imfy2-1)< Isz2)
	{
		IplImage* img2;
		yres=imfy2;
		xres=Isz1-imfx2;
		*xadp=(Pfx1-xres)*(-1);
		*yadp=floor(imfy2)-Pfy1+1;
		cvSetImageROI(frame, cvRect(xres-imfx2,0,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
//		rcrop_selection(frame,&crop_scene,xres-imfx2,xres+imfx2-1,0,yres+imfy2-1);
	}

	//Lower right corner
	if ((Pfx1-imfx2)>0 && (Pfx1+imfx2-1)>=Isz1 && (Pfy1-imfy2)>0 && (Pfy1+imfy2-1)>= Isz2)
	{
		IplImage* img2;
		yres=Isz2-imfy2;
		xres=Isz1-imfx2;
		*xadp=(Pfx1-xres)*(-1);
		*yadp=(Pfy1-yres)*(-1);
		cvSetImageROI(frame, cvRect(xres-imfx2,yres-imfy2,co2,re2));
		img2=cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		//crop_scene = cvCreateImage(cvGetSize(frame),frame->depth,frame->nChannels);
		cvCopy(frame,img2, NULL);
		cvResetImageROI(frame);
		cvScale(img2,crop_scene,1/255.,0);
//		rcrop_selection(frame,&crop_scene,xres-imfx2,xres+imfx2-1,yres-imfy2,yres+imfy2-1);
	}

	//Crop Imf
	//Normal case
	if ((x1-imfx2)>0 && (x1+imfx2-1) < Isz1 && (y1-imfy2)>0 && (y1+imfy2-1) < Isz2)
	{
		*recc=1;
		IplImage* img;
		cvSetImageROI(prevf, cvRect(x1-imfx2,y1-imfy2,Imfr->width,Imfr->height));
		img=cvCreateImage(cvGetSize(prevf),prevf->depth,prevf->nChannels);
		cvCopy(prevf,img, NULL);
		cvResetImageROI(prevf);
		cvScale(img,Imfr,1/255.,0);
//		rcrop_selection(prevf,&Imfr,x1-imfx2,x1+imfx2-1,y1-imfy2,y1+imfy2-1);
	}
	
	//Overflow on the upper side (x1-imfx2)<0
	if ((x1-imfx2)<=0 && (x1+imfx2-1) < Isz1 && (y1-imfy2)>0 && (y1+imfy2-1) < Isz2)
	{
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		sz1=abs((x1-imfx2+re2)-(x1+imfx2+re2-1));
		sz2=abs((y1-imfy2)-(y1+imfy2-1));
		cresx=ceil(x1-imfx2)*(-1)+1;
		cvSetImageROI(prevf, cvRect(x1-imfx2+re2+cresx,y1-imfy2,co2,re2-cresx));
		cvSetImageROI(Imfr, cvRect(cresx,0,co2,re2-cresx));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
//		rcrop_selection(prevf,&Imfr,x1-imfx2+re2+cresx,x1+imfx2+re2-1,y1-imfy2,y1+imfy2-1);
		//Mirror the elements near to the edge into the empty elements
		for (int i = 1; i <= cresx; ++i)
		{
			for (int j = 0; j < sz2; ++j)
			{
				Imfr->imageData[(cresx-i)*sz2+j]=Imfr->imageData[(cresx-1+i)*sz2+j];
			}
		}
		*recc=0;
	}

	//overflow on the lower side (x1+imfx2-1)>480
	if ((x1-imfx2)>0 && (x1+imfx2-1)>=Isz1 && (y1-imfy2)>0 && (y1+imfy2-1) < Isz2)
	{	
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		sz1=abs((x1-imfx2)-(x1+imfx2-1));
		sz2=abs((y1-imfy2)-(y1+imfy2-1));
		cresx=ceil(x1+imfx2-1-Isz1);
		cvSetImageROI(prevf, cvRect(x1-imfx2,y1-imfy2,co2,re2-cresx));
		cvSetImageROI(Imfr, cvRect(0,0,co2,re2-cresx));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
		//rcrop_selection(prevf,&Imfr,x1-imfx2,x1+imfx2-1-cresx,y1-imfy2,y1+imfy2-1);

		//mirror
		for (int i = 0; i < cresx; ++i)
		{
			for (int j = 0; j < sz2; ++j)
			{
				Imfr->imageData[(sz2-cresx+i)*sz2+j]=Imfr->imageData[(sz2-cresx-1-i)*sz2+j];
			}
		}
		*recc=0;
	}
	//left side(y1-imfy2)<0
	if ((x1-imfx2)>0 && (x1+imfx2-1)<Isz1 && (y1-imfy2)<=0 && (y1+imfy2-1) < Isz2)
	{
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		sz1=abs((x1-imfx2)-(x1+imfx2-1));
		sz2=abs((y1-imfy2+co2)-(y1+imfy2+co2-1));
		cresy=ceil(y1-imfy2)*(-1)+1;
		cvSetImageROI(prevf, cvRect(x1-imfx2,y1-imfy2+co2+cresy,co2-cresy,re2));
		cvSetImageROI(Imfr, cvRect(0,cresy,co2-cresy,re2));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
//		rcrop_selection(prevf,&Imfr,x1-imfx2,x1+imfx2-1,y1-imfy2+co2+cresy,y1+imfy2+co2-1);
		//mirror
		for (int j = 1; j <= cresy; ++j)
		{
			for (int i = 0; i < sz1; ++i)
			{
				Imfr->imageData[i*sz2+(cresy-j)]=Imfr->imageData[i*sz2+(cresy-1+j)];
			}
		}
		*recc=0;
	}
	//right side (y1+imfy2-1)> (640+sv*2)
	if ((x1-imfx2)>0 && (x1+imfx2-1)<Isz1&& (y1-imfy2)>0 && (y1+imfy2-1)>= Isz2)
	{
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		sz1=abs((x1-imfx2)-(x1+imfx2-1));
		sz2=abs((y1-imfy2)-(y1+imfy2-1));
		cresy=ceil(y1+imfy2-1-Isz2);
		cvSetImageROI(prevf, cvRect(x1-imfx2,y1-imfy2,co2-cresy,re2));
		cvSetImageROI(Imfr, cvRect(0,0,co2-cresy,re2));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
//		rcrop_selection(prevf,&Imfr,x1-imfx2,x1+imfx2-1,y1-imfy2,y1+imfy2-1-cresy);
		//mirror
		for (int j = 1; j <= cresy; ++j)
		{
			for (int i = 0; i < sz1; ++i)
			{
				Imfr->imageData[i*sz2+(sz2-cresy+j)]=Imfr->imageData[i*sz2+(sz2-cresy-1-j)];
			}
		}
		*recc=0;
	}

	//upper left corner
	if ((x1-imfx2)<=0 && (x1+imfx2-1)<Isz1 && (y1-imfy2)<=0 && (y1+imfy2-1)< Isz2)
	{
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		xres=ceil(imfx2)+x1;  
		yres=ceil(imfy2)+y1;
		cresx=ceil(x1-imfx2)*(-1)+1;
		cresy=ceil(y1-imfy2)*(-1)+1;
		cvSetImageROI(prevf, cvRect(xres-imfx2+cresx,yres-imfy2+cresy,co2-cresy,re2-cresx));
		cvSetImageROI(Imfr, cvRect(cresx,cresy,co2-cresy,re2-cresx));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
//		rcrop_selection(prevf,&Imfr,xres-imfx2+cresx,xres+imfx2-1,yres-imfy2+cresy,yres+imfy2-1);
		//mirror
		for (int i = 1; i <= cresx; ++i)
		{
			for (int j = 0; j < sz2; ++j)
			{
				Imfr->imageData[(cresx-i)*sz2+j]=Imfr->imageData[(cresx-1+i)*sz2+j];
			}
		}
		//mirror
		for (int j = 1; j <= cresy; ++j)
		{
			for (int i = 0; i < sz1; ++i)
			{
				Imfr->imageData[i*sz2+(cresy-j)]=Imfr->imageData[i*sz2+(cresy-1+j)];
			}
		}
		*recc=0;
	}
	//Upper right corner
	if ((x1-imfx2)<=0 && (x1+imfx2-1)<Isz1 && (y1-imfy2)>0 && (y1+imfy2-1)>= Isz2)
	{
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		xres=ceil(imfx2)+x1;
		//Imfr=lastim(xres-imfx2:xres+imfx2-1,y1-imfy2:y1+imfy2-1);
		cresx=ceil(x1-imfx2)*(-1)+1;
		cresy=ceil(y1+imfy2-1-Isz2);
		cvSetImageROI(prevf, cvRect(xres-imfx2+cresx,y1-imfy2,co2-cresy,re2-cresx));
		cvSetImageROI(Imfr, cvRect(cresx,0,co2-cresy,re2-cresx));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
//		rcrop_selection(prevf,&Imfr,xres-imfx2+cresx,xres+imfx2-1,y1-imfy2,y1+imfy2-1-cresy);
		//Mirror
		for (int i = 1; i <= cresx; ++i)
		{
			for (int j = 0; j < sz2; ++j)
			{
				Imfr->imageData[(cresx-i*sz2+j)]=Imfr->imageData[(cresx-1+i)*sz2+j];
			}
		}
		//Mirror
		for (int j = 1; j <= cresy; ++j)
		{
			for (int i = 0; i < sz1; ++i)
			{
				Imfr->imageData[i*sz2+(sz2-cresy+j)]=Imfr->imageData[i*sz2+(sz2-cresy-1-j)];
			}
		}
		*recc=0;
	}
	//Lower left corner
	if ((x1-imfx2)>0 && (x1+imfx2-1)>=Isz1 && (y1-imfy2)<=0 && (y1+imfy2-1)< Isz2)
	{
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		yres=ceil(imfy2)+y1;
		cresy=ceil(y1-imfy2)*(-1)+1;
		cresx=ceil(x1+imfx2-1-Isz1);
		cvSetImageROI(prevf, cvRect(x1-imfx2,yres-imfy2+cresy,co2-cresy,re2-cresx));
		cvSetImageROI(Imfr, cvRect(0,cresy,co2-cresy,re2-cresx));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
//		rcrop_selection(prevf,&Imfr,x1-imfx2,x1+imfx2-1-cresx,yres-imfy2+cresy,yres+imfy2-1);
		//Mirror
		for (int i = 0; i < cresx; ++i)
		{
			for (int j = 0; j < sz2; ++j)
			{
				Imfr->imageData[(sz1-cresx+i)*sz2+j]=Imfr->imageData[(sz1-cresx-1-i)*sz2+j];
			}
		}
		//Mirror
		for (int j = 1; j <= cresy; ++j)
		{
			for (int i = 0; i < sz1; ++i)
			{
				Imfr->imageData[i*sz2+(cresy-j)]=Imfr->imageData[i*sz2+(cresy-1+j)];
			}
		}
		*recc=0;
	}
	//Lower right corner
	if ((x1-imfx2)>0 && (x1+imfx2-1)>=Isz1 && (y1-imfy2)>0 && (y1+imfy2-1)>= Isz2)
	{
		IplImage* img;
		img=cvCreateImage(cvGetSize(Imfr),prevf->depth,prevf->nChannels);
		//Imfr=lastim(x1-imfx2:x1+imfx2-1,y1-imfy2:y1+imfy2-1);
		cresx=ceil(x1+imfx2-1-Isz1);
		cresy=ceil(y1+imfy2-1-Isz2);
		cvSetImageROI(prevf, cvRect(x1-imfx2,y1-imfy2,co2-cresy,re2-cresx));
		cvSetImageROI(Imfr, cvRect(0,0,co2-cresy,re2-cresx));
		cvCopy(prevf,img, NULL);
		cvScale(img,Imfr,1/255.,0);
		cvResetImageROI(prevf);
		cvResetImageROI(Imfr);
//		rcrop_selection(prevf,&Imfr,x1-imfx2,x1+imfx2-1,y1-imfy2,y1+imfy2-1);
		//Mirror
		for (int i = 0; i < cresx; ++i)
		{
			for (int j = 0; j < sz2; ++j)
			{
				Imfr->imageData[(sz1-cresx+i)*sz2+j]=Imfr->imageData[(sz1-cresx-1-i)*sz2+j];
			}
		}
		//Mirror
		for (int j = 1; j <= cresy; ++j)
		{
			for (int i = 0; i < sz1; ++i)
			{
				Imfr->imageData[i*sz2+(sz2-cresy+j)]=Imfr->imageData[i*sz2+(sz2-cresy-1-j)];
			}
		}
		*recc=0;
	}
}
/*
	(m-min)/(max-min)
*/