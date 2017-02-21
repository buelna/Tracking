#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <time.h>
#include <stdio.h>
#include <ctype.h>
#include "../includes/tracking_utils.h"
#include "../includes/matrix_operations.h"
#include "../includes/fft_utils.h"
#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <complex.h>
static void help(void)
{
    printf(
            "\nThis program demonstrated the use of motion templates -- basically using the gradients\n"
            "of thresholded layers of decaying frame differencing. New movements are stamped on top with floating system\n"
            "time code and motions too old are thresholded away. This is the 'motion history file'. The program reads from the camera of your choice or from\n"
            "a file. Gradients of motion history are used to detect direction of motoin etc\n"
            "Usage :\n"
            "./motempl [camera number 0-n or file name, default is camera 0]\n"
            );
}

// parameters:
//  img - input video frame
//  dst - resultant motion picture
//  args - optional parameters
int main(int argc, char** argv)
{
    IplImage *curr_selection=0,*Ipltmp1,stub,stub1,*Ipltmp2,stub2,*Ipltmp3,stub3;
    CvCapture* capture = 0;
    //help();
    double *DCTT,**gauss_win,**gauss_winL,sig;//array to store all the DC's values
    int xcenter,ycenter,frag_lines,frag_cols,lenc;
    int half_lines,half_cols,frame_szx,frame_szy,true_class_counter;
    int counter,true_template_counter,found,local_search_counter,nframes,tmpf;
    int tsearch,build_filters,out_of_bounds,xadp,yadp,frame_counter,start_frame;
    double DC,rho_old,weighing_filters,rho_x,rho_y,vari,pot1,pot2,**s,maxs,meanImf_Old,Zden,Zden2;
    double *Z1,*Z2,pearson,med;
    double complex ** fftmp,**Pn,**H,**F,**T,**Q,*c,*sdfilter,**sdTmpMat,**sdTmpMati,**Hold,**filter,**NUM,**num,**conjH,**conjF,**multmp;
    double complex **den1,**den2,stmp;
    /*
        Open file containing the video sequence or read from camera
    */
    if( argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
        capture = cvCaptureFromCAM( argc == 2 ? argv[1][0] - '0' : 0 );
    else if( argc == 2 )
        capture = cvCaptureFromFile( argv[1] );
    /*
        Read object position from file
    */
    FILE * in_file;
    in_file = fopen("coordenadas.txt", "r");
    if ( !in_file ) {
        return 0;
    }
    fscanf(in_file,"%d %d %d %d",&xcenter,&ycenter,&frag_lines,&frag_cols);
    fclose(in_file);
    /*
        Initialize parameters
    */
    tmpf=0;
    sig=0.0;//noise estimation
    half_lines=frag_lines/2;//half the lines of the selection area
    half_cols=frag_cols/2;//half columns of the selection area
    lenc=frag_cols*frag_lines;
    DC=1.0;//discrimination capacity
    frame_szx=480;//frame size (heigh)
    frame_szy=640;//frame size (width)
    counter=0;//counter for prediction stage
    true_template_counter=1;//counter for true templates
    found=1;//flag to indicate that the object was found
    local_search_counter=1;//counts entries to local search
    tsearch=0;//flag to enter total search
    build_filters=0;//flag to enable construction of filters for local stage
    out_of_bounds=1;//flag to indicate if the croped area is out of the limits of the frame
    xadp=0;//adaptation for x in case of out of bounds
    yadp=0;//adaptation for y in case of out of bounds
    rho_old=0.1;
    true_class_counter=1;//counter for templates in true class
    weighing_filters=0.125;//weighing of current and past filter
    /*
        initialize matrix for GMF templates and matrix for object templates
    */
        create_cmatrix(&T, 9,lenc);
        create_cmatrix(&Q, 9,lenc);
        create_cmatrix(&sdTmpMat, 9,9);
        create_cmatrix(&sdTmpMati, 9,9);
        sdfilter=(double complex *)malloc(lenc*sizeof(double complex));
        c=(double complex*)malloc(9*sizeof(double complex));
    /*
        Initialize parameters for local search
    */
        int localLines,localCols;
        double **ks,**ls,**kl,**ll,**Imfg,**R;
        localLines=frag_lines/2;
        localLines=(localLines*2)+frag_lines;
        localCols=2*frag_cols-1;
    /*
        Read the first frame of the video
    */
    start_frame=0;
    frame_counter=start_frame;
    IplImage* curr_frame =0;
    curr_frame=cvQueryFrame( capture );
    if( !curr_frame )
        return 1;
    frame_szx=curr_frame->height;//frame size (heigh)
    frame_szy=curr_frame->width;//frame size (width)
    IplImage* cf_gray=0;
    cf_gray=cvCreateImage(cvGetSize(curr_frame),curr_frame->depth,1);

    /*
        Convert the current frame to grayscale
    */
    cvCvtColor(curr_frame,cf_gray,CV_BGR2GRAY);
    //frame_counter++;
    /*
        Crop selection
    */
    cvNamedWindow( "Motion", 0 );
    cvNamedWindow( "B", 0 );
    cvSetImageROI(cf_gray, cvRect(ycenter-half_cols-1,xcenter-half_lines,frag_cols,frag_lines));
    curr_selection = cvCreateImage(cvGetSize(cf_gray),cf_gray->depth,cf_gray->nChannels);
    cvReleaseImage(&curr_selection);
    curr_selection=cvCloneImage(cf_gray);
    cvResetImageROI(cf_gray);
    IplImage* img32=cvCreateImage(cvGetSize(curr_selection),IPL_DEPTH_32F,1);
    cvScale(curr_selection,img32,1/255.,0);
    cvReleaseImage(&curr_selection);
    curr_selection=cvCloneImage(img32);
    /*
        Creating utils for covariance function and gaussian windows aplication
    */
    create_matrix(&gauss_winL, localCols,localLines);//create matrix to store gaussian window same size as curr_selection
    gaussian_2Dwindow(gauss_winL,localLines,localCols);
    create_matrix(&gauss_win, frag_cols,frag_lines);//create matrix to store gaussian window same size as curr_selection
    gaussian_2Dwindow(gauss_win,frag_lines,frag_cols);
    create_matrix(&ks, frag_cols,frag_lines);
    create_matrix(&ls, frag_cols,frag_lines);
    create_matrix(&kl, localCols,localLines);
    create_matrix(&ll, localCols,localLines);
    create_matrix(&Imfg, localCols,localLines);
    create_matrix(&R, frag_cols,frag_lines);
    create_cmatrix(&F, frag_cols,frag_lines);
    create_cmatrix(&conjH, frag_cols,frag_lines);
    create_cmatrix(&conjF, frag_cols,frag_lines);
    create_cmatrix(&multmp, frag_cols,frag_lines);
    create_cmatrix(&den1, frag_cols,frag_lines);
    create_cmatrix(&den2, frag_cols,frag_lines);
    create_matrix(&s, frag_cols,frag_lines);
    create_cmatrix(&NUM, frag_cols,frag_lines);
    create_cmatrix(&num, frag_cols,frag_lines);
    create_cmatrix(&filter,frag_cols,frag_lines);
    create_cmatrix(&fftmp,frag_cols,frag_lines);
    create_cmatrix(&Hold,frag_cols,frag_lines);
    create_cmatrix(&H,frag_cols,frag_lines);
    create_cmatrix(&Pn,frag_cols,frag_lines);
    for (int i = 0; i < frag_lines; ++i)
    {
        for (int j = 0; j < frag_cols; ++j)
        {
            ks[i][j]=j+1;
            ls[i][j]=i+1;
        }
    }
    for (int i = 0; i < localLines; ++i)
    {
        for (int j = 0; j < localCols; ++j)
        {
            kl[i][j]=j+1;
            ll[i][j]=i+1;
            Imfg[i][j]=0;
        }
    }
    /*
        applying the gaussian window on the current selection to smooth edges and background effects
    */
    cvShowImage( "Motion", cf_gray );
    CvMat* tempMat=cvCreateMat(img32->height,img32->width,CV_32FC1);
    CvMat **imgsTrue,**imgsTrue1;
    CvMat** q;
    imgsTrue=(CvMat**)malloc(9*sizeof(CvMat*));
    imgsTrue1=(CvMat**)malloc(9*sizeof(CvMat*));
    q=(CvMat**)malloc(9*sizeof(CvMat*));
    for (int i = 0; i < 9; ++i)
    {
        imgsTrue[i]=cvCreateMat(img32->height,img32->width,CV_32FC1);
        imgsTrue1[i]=cvCreateMat(img32->height,img32->width,CV_32FC1);
        q[i]=cvCreateMat(img32->height,img32->width,CV_32FC1);
        c[i]=1;
    }
    CvMat* M=cvCreateMat(img32->height,img32->width,CV_32FC1);
    CvMat* prev_selection=cvCreateMat(img32->height,img32->width,CV_32FC1);
    CvMat* M1=cvCreateMat(img32->height,img32->width,CV_32FC1);
    CvMat* scene=cvCreateMat(img32->height,img32->width,CV_32FC1);
    CvMat* cSelection=cvCreateMat(img32->height,img32->width,CV_32FC1);
    IplImage *crop=0;
    IplImage *prev_frame= 0;
    cvReleaseMat(&M);
    M=cvCloneMat(cvGetMat(img32,tempMat,0,0));
    cvReleaseImage(&crop);
    crop=cvCloneImage(cvGetImage(M,&stub));
    mat2grayM(M,frag_lines,frag_cols);
    matrix_multiplication(gauss_win,M,cSelection,frag_lines,frag_cols);
    
    
if( capture )
{
    /*
        Prediction initialization values
    */
    int Pfx1,Pfy1,tx[5]={-2,-1,0,1,2};
    Pfx1=xcenter;
    Pfy1=ycenter;
    int ssqx=10,sqqx=34,numpred=5;
    int ex[5]={0,0,0,0,0},ey[5]={0,0,0,0,0};
    Z1=(double *)malloc(frag_lines*frag_cols*sizeof(double));
    Z2=(double *)malloc(frag_lines*frag_cols*sizeof(double));
    for(;;)
    {
        //store las object template
        cvReleaseImage(&prev_frame);
        prev_frame=cvCloneImage(cf_gray);//-> lastim
        cvReleaseMat(&prev_selection);
        prev_selection=cvCloneMat(cSelection);//->Imf_Old
        mat2grayM(prev_selection,frag_lines,frag_cols);//fix function within functions as well
        curr_frame = cvQueryFrame( capture );//read next frame
        if( !curr_frame )
            break;
        cvCvtColor(curr_frame,cf_gray,CV_BGR2GRAY);
        if (!tsearch)
        {
            if (found)
            {
                /*
                    Prediction stage from the first 5 positions onwards
                    in x and y
                */
                if (frame_counter>4)
                {
                    //counter=0;//1
                    for (int i = 0; i < 4; ++i)
                    {
                        ex[i]=ex[i+1];
                        ey[i]=ey[i+1];
                    }
                    ex[4]=xcenter;
                    ey[4]=ycenter;
                    prediction(ex,ey,numpred,tx,ssqx,sqqx,&Pfx1,&Pfy1);

                }
                else
                {
                    ex[counter]=xcenter;
                    ey[counter]=ycenter;
                    Pfx1=xcenter;
                    Pfy1=ycenter;
                    counter++;
                }
                /*
                    Crop the fragment from the current scene and the template of the object from the prev frame
                    considering the image boundaries
                */

                if (frame_counter>start_frame)
                {
normal_crop(xcenter,ycenter,frame_szx,frame_szy,half_lines,half_cols,cf_gray,prev_frame,frag_lines,frag_cols,
frag_lines,frag_cols,Pfx1,Pfy1,curr_selection,crop,&xadp,&yadp,&out_of_bounds);
                }

                cvReleaseMat(&M);
                M=cvCloneMat(cvGetMat(curr_selection,cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                mat2grayM(M,frag_lines,frag_cols);
                matrix_multiplication(gauss_win,M,cSelection,frag_lines,frag_cols);
                sig=in_noise_est(cSelection,frag_lines,frag_cols);
                est_rho(&cSelection->data.fl,frag_lines,frag_cols,&rho_x);
                rho_x=rho_x-(35*sig);
                //rho_x= 0.3>rho_x?0.3:rho_x;
                rho_x=weighing_filters*rho_x+(1-weighing_filters)*rho_old;
                rho_old=rho_x;
                rho_y=rho_x;
                rho_x=0.2;
                vari=var(&cSelection->data.fl,frag_lines,frag_cols);
                for (int i = 0; i < frag_lines; ++i)
                {
                    for (int j = 0; j < frag_cols; ++j)
                    {
                        pot1=((ks[i][j]-frag_cols/2)/frag_lines);
                        pot2=((ls[i][j]-frag_lines/2)/frag_cols);
                        R[i][j]=vari*pow(rho_x,cabs(pot1))*pow(rho_y,cabs(pot2));
                    }
                }
                rfft2(R,fftmp,frag_lines,frag_cols);
                cmat_abs(fftmp,frag_lines,frag_cols);
                fftshift(&fftmp,frag_lines,frag_cols);
                for (int i = 0; i < frag_lines; ++i)
                {
                    for (int j = 0; j < frag_cols; ++j)
                    {
                        Pn[i][j]=sig+fftmp[i][j];
                    }
                }
                srfft2(cSelection,fftmp,frag_lines,frag_cols);
                fftshift(&fftmp,frag_lines,frag_cols);
                cmatrix_div(fftmp,Pn,H,frag_lines,frag_cols);
                ifftshift(&H,frag_lines,frag_cols);
                dcmatrix_multiplication(gauss_win,H,fftmp,frag_lines,frag_cols);
                ifft2Msp(fftmp,M,M1,frag_lines,frag_cols);

                /*
                    Rotate images in both directions up to 12 degrees
                */
                cvReleaseMat(&imgsTrue[0]);
                cvReleaseMat(&imgsTrue1[0]);
                cvReleaseMat(&q[0]);
                Ipltmp1=cvCloneImage(cvGetImage(M,&stub1));
                Ipltmp3=cvCloneImage(cvGetImage(M1,&stub3));
                Ipltmp2=cvCloneImage(cvGetImage(cSelection,&stub2));
                imgsTrue[0]=cvCloneMat(M);
                imgsTrue1[0]=cvCloneMat(M1);
                q[0]=cvCloneMat(cSelection);
                for (int i = 0; i < 4; ++i)
                {
                	cvReleaseMat(&imgsTrue[i+1]);
                	cvReleaseMat(&imgsTrue[i+5]);
                	cvReleaseMat(&imgsTrue1[i+1]);
                	cvReleaseMat(&imgsTrue1[i+5]);
                	cvReleaseMat(&q[i+1]);
                	cvReleaseMat(&q[i+5]);
                    imgsTrue[i+1]=cvCloneMat(cvGetMat(rotateImage(Ipltmp1,3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                    imgsTrue[i+5]=cvCloneMat(cvGetMat(rotateImage(Ipltmp1,-3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                    imgsTrue1[i+1]=cvCloneMat(cvGetMat(rotateImage(Ipltmp3,3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                    imgsTrue1[i+5]=cvCloneMat(cvGetMat(rotateImage(Ipltmp3,-3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                    q[i+1]=cvCloneMat(cvGetMat(rotateImage(Ipltmp2,3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                    q[i+5]=cvCloneMat(cvGetMat(rotateImage(Ipltmp2,-3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                }
                cvReleaseImage(&Ipltmp1);
                cvReleaseImage(&Ipltmp2);
                cvReleaseImage(&Ipltmp3);
                /*
                    pass each image to its column form and generate the templates
                */
                for (int i = 0; i < 9; ++i)
                {
                    srfft2M(imgsTrue[i],imgsTrue1[i],fftmp,frag_lines,frag_cols);
                    creshape(frag_lines,frag_cols,fftmp,T,i);

                    srfft2(q[i],fftmp,frag_lines,frag_cols);
                    creshape(frag_lines,frag_cols,fftmp,Q,i);
                }

                t1matXmat(Q,T,sdTmpMat,lenc,9,lenc,9);
                Cinverse(sdTmpMat,sdTmpMati,9);
                matXmat(T,sdTmpMati,Q,lenc,9,9,9);
                matXvec(Q,c,sdfilter,lenc,9,9,1);
                creshapeB(frag_lines,frag_cols, sdfilter,fftmp);

                if (frame_counter>start_frame)
                {
                    for (int i = 0; i < frag_lines; ++i)
                    {
                        for (int j = 0; j < frag_cols; ++j)
                        {
                            fftmp[i][j]=(weighing_filters*fftmp[i][j])+((1-weighing_filters)*Hold[i][j]);
                        }
                    }
                }
                matCpy(frag_lines,frag_cols,fftmp,Hold,1);//store current filter
                /*
                    pre-processing of scene crop and filter
                */
                
                cvReleaseMat(&scene);
                scene=cvCloneMat(cvGetMat(crop,cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
                
                matrix_multiplication(gauss_win,scene,scene,frag_lines,frag_cols);
                med=avg(&scene->data.fl,frag_lines,frag_cols);
                subsMat(frag_lines,frag_cols,scene,scene,med);
                
                ifft2(fftmp,filter,frag_lines,frag_cols);
                subsMatrix(frag_lines,frag_cols,filter,filter,cMatMean(frag_lines,frag_cols,filter));
                
                srfft2(scene,F,frag_lines,frag_cols);
                
                fftshift(&F,frag_lines,frag_cols);
                fft2(filter,fftmp,frag_lines,frag_cols);//fftmp=H ///filter?
                fftshift(&fftmp,frag_lines,frag_cols);
                
                cconj(fftmp,conjH,frag_lines,frag_cols);
                cconj(F,conjF,frag_lines,frag_cols);
                cmatrix_multiplication(F,conjH,NUM,frag_lines,frag_cols);
                ifft2(NUM,num,frag_lines,frag_cols);
                ifftshift(&num,frag_lines,frag_cols);
                /*
                    Normalizing correlation plane
                */
                cmatrix_multiplication(F,conjF,multmp,frag_lines,frag_cols);
                
                ifft2(multmp,den1,frag_lines,frag_cols);
                ifftshift(&den1,frag_lines,frag_cols);
                cmatrix_multiplication(fftmp,conjH,multmp,frag_lines,frag_cols);
                ifft2(multmp,den2,frag_lines,frag_cols);
                ifftshift(&den2,frag_lines,frag_cols);
            
                for (int i = 0; i < frag_lines; ++i)
                {
                    for (int j = 0; j < frag_cols; ++j)
                    {
                        stmp=cabs(creal(num[i][j]/(0.1+den1[i][j]*den2[i][j])));
                        s[i][j]=stmp*stmp;
                    }
                }
                DC=calcDCfast(s,frag_lines,frag_cols);
                maxs=max(s,frag_lines,frag_cols);
               
                for (int i = 0; i < frag_lines; ++i)
                {
                    for (int j = 0; j < frag_cols; ++j)
                    {
                        s[i][j]=s[i][j]/maxs;
                    }
                }
                cvShowImage("B",cSelection);
                mat2gray(&s,frag_lines,frag_cols);
                
                meanImf_Old=avg(&prev_selection->data.fl,frag_lines,frag_cols);
                subsMat(frag_lines,frag_cols,prev_selection,prev_selection,meanImf_Old);
                Zden=0;
                Zden2=0;
                cvReleaseMat(&M);
                M=cvCloneMat(cSelection);
                mat2grayM(M,frag_lines,frag_cols);
                meanImf_Old=avg(&M->data.fl,frag_lines,frag_cols);
                subsMat(frag_lines,frag_cols,M,M,meanImf_Old);
                for (int i = 0; i < frag_lines; ++i)
                {
                    for (int j = 0; j < frag_cols; ++j)
                    {
                       Zden+=prev_selection->data.fl[i*frag_cols+j]*prev_selection->data.fl[i*frag_cols+j]; 
                       Zden2+=M->data.fl[i*frag_cols+j]*M->data.fl[i*frag_cols+j]; 
                    }
                }
                Zden=sqrt(Zden);
                Zden2=sqrt(Zden2);
                for (int i = 0; i < frag_lines; ++i)
                {
                    for (int j = 0; j < frag_cols; ++j)
                    {
                       Z1[i*frag_cols+j]=prev_selection->data.fl[i*frag_cols+j]/Zden;
                       Z2[i*frag_cols+j]=M->data.fl[i*frag_cols+j]/Zden2;
                    }
                }
                pearson=0;
                for (int i = 0; i < lenc; ++i)
                {
                    pearson+=Z1[i]*Z2[i];
                }
                if (DC>0.4)
                {
                    findMax(s,frag_lines,frag_cols,&xcenter,&ycenter);
                    xcenter=Pfx1-1+yadp+(xcenter-half_lines);//yadp
                    ycenter=Pfy1-1+xadp+(ycenter-half_cols);//xadp
                    tmpf=1;
                }
                
                if (pearson>=0.4)
                {
                    if (out_of_bounds==1)
                    {
                        printf("%f\n",DC);
                        if (DC>0.4)
                        {
                            cvRectangle(curr_frame,cvPoint(ycenter-(half_cols/2)-1,xcenter-(half_lines/2)-1),cvPoint(ycenter+(half_cols/2)-1,xcenter+(half_lines/2-1)),cvScalar(255,40,40,0),3,8,0);
                            //cvRectangle(curr_frame,cvPoint(ycenter-(half_cols)-1,xcenter-(half_lines)-1),cvPoint(ycenter+(half_cols)-1,xcenter+(half_lines-1)),cvScalar(255,40,40,0),3,8,0);
                            cvShowImage("Motion", curr_frame );
                        }
                        else
                        {
                            cvRectangle(curr_frame,cvPoint(ycenter-(half_cols/2)-1,xcenter-(half_lines/2)-1),cvPoint(ycenter+(half_cols/2)-1,xcenter+(half_lines/2)-1),cvScalar(40,255,40,0),3,8,0);
                            //cvRectangle(curr_frame,cvPoint(ycenter-(half_cols)-1,xcenter-(half_lines)-1),cvPoint(ycenter+(half_cols)-1,xcenter+(half_lines)-1),cvScalar(40,255,40,0),3,8,0);
                            cvShowImage("Motion", curr_frame );
                            //found=0;
                            tmpf=1;
                        }
                    }
                    else
                    {
                        printf("%f xadp:%d\tyadp:%d\n",DC,xadp,yadp );
                    }
                }
            }
            if( cvWaitKey(10) >= 0 )
                break;
        }
        frame_counter++;
    }
    cvReleaseCapture( &capture );
    cvDestroyWindow( "Motion" );
    return 0;
}
}