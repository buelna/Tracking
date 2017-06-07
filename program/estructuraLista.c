typedef struct node {
    struct node * next;
    int xcenter,ycenter,xadp,yadp,out_of_bounds,frame_counter,Pfx1,Pfy1;
    double **R,**T,**Q,DC,**s,sig;
    double complex **fftmp,**Hold,**den1,**den2**num;
    IplImage *curr_frame,*cf_gray*prev_frame,*curr_selection,*crop;
    CvMat *cSelection,*M,**imgsTrue,**q;
} node_t;
///////////////////
void * readFrame(void *args);
void * prediction(void *args);
void * crop(void *args);
void * ThreadR(void *args);
void * ThreadPn(void *args);
void * rotation(void *args);
void * Reshape(void *args);
void * Hfilter(void *args);
void * preprocessing(void *args);
void * ThreadS(void *args);
void * Find(void *args);
void * Show(void *args);
///////////
struct sched_param paramRead, paramPredict, paramCrop,paramR,paramPn,paramRotation,paramReshape,paramH;
struct sched_param paramProcessing,paramS,paramFind,paramShow;
pthread_t thPredict,thRead,thCrop,thR,thPn,thRotation,thReshape,thH,thProcessing,thS,thFind,thShow;
pthread_attr_t attrRead,attrPredict,attrCrop,attrR,attrPn,attrRotation,attrReshape,attrH,attrProcessing,attrS,attrFind,attrShow;
////////////////
status = pthread_attr_init(&attrRead);
if (status != 0) {
	perror("Init attrRead");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrRead,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrRead");
	exit(2);
}

status = pthread_attr_init(&attrPredict);
if (status != 0) {
	perror("Init attrPredict");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrPredict,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrPredict");
	exit(2);
}

status = pthread_attr_init(&attrCrop);
if (status != 0) {
	perror("Init attrCrop");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrCrop,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrCrop");
	exit(2);
}

status = pthread_attr_init(&attrR);
if (status != 0) {
	perror("Init attrR");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrR,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrR");
	exit(2);
}

status = pthread_attr_init(&attrPn);
if (status != 0) {
	perror("Init attrPn");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrPn,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrPn");
	exit(2);
}

status = pthread_attr_init(&attrRotation);
if (status != 0) {
	perror("Init attrRotation");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrRotation,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrRotation");
	exit(2);
}

status = pthread_attr_init(&attrReshape);
if (status != 0) {
	perror("Init attrReshape");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrReshape,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrReshape");
	exit(2);
}

status = pthread_attr_init(&attrH);
if (status != 0) {
	perror("Init attrH");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrH,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrH");
	exit(2);
}

status = pthread_attr_init(&attrProcessing);
if (status != 0) {
	perror("Init attrProcessing");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrProcessing,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrProcessing");
	exit(2);
}

status = pthread_attr_init(&attrS);
if (status != 0) {
	perror("Init attrS");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrS,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrS");
	exit(2);
}

status = pthread_attr_init(&attrFind);
if (status != 0) {
	perror("Init attrFind");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrFind,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrFind");
	exit(2);
}

status = pthread_attr_init(&attrShow);
if (status != 0) {
	perror("Init attrShow");
	exit(2);
}
status = pthread_attr_setinheritsched(&attrShow,PTHREAD_EXPLICIT_SCHED);
if (status != 0) {
	perror("EXPLICIT attrShow");
	exit(2);
}
///////////////set policy
status = pthread_attr_setschedpolicy(&attrRead, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrRead.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrPredict, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrPredict.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrCrop, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrCrop.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrR, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrR.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrPn, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrPn.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrRotation, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrRotation.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrReshape, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrReshape.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrH, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrH.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrProcessing, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrProcessing.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrS, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrS.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrFind, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrFind.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrFind, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrFind.\n");
	exit(1);
}

status = pthread_attr_setschedpolicy(&attrShow, SCHED_FIFO);
if (status != 0) {
	perror("Unable to set SCHED_FIFO policy to attrShow.\n");
	exit(1);
}
///////set params
paramRead.sched_priority = 99;
status = pthread_attr_setschedparam(&attrRead, &paramRead);
if (status != 0) {
	perror("Unable to set params to attrRead.\n");
}
paramPredict.sched_priority = 99;
status = pthread_attr_setschedparam(&attrPredict, &paramPredict);
if (status != 0) {
	perror("Unable to set params to attrPredict.\n");
}
paramCrop.sched_priority = 99;
status = pthread_attr_setschedparam(&attrCrop, &paramCrop);
if (status != 0) {
	perror("Unable to set params to attrCrop.\n");
}
paramR.sched_priority = 99;
status = pthread_attr_setschedparam(&attrR, &paramR);
if (status != 0) {
	perror("Unable to set params to attrR.\n");
}
paramPn.sched_priority = 99;
status = pthread_attr_setschedparam(&attrPn, &paramPn);
if (status != 0) {
	perror("Unable to set params to attrPn.\n");
}
paramRotation.sched_priority = 99;
status = pthread_attr_setschedparam(&attrRotation, &paramRotation);
if (status != 0) {
	perror("Unable to set params to attrRotation.\n");
}
paramReshape.sched_priority = 99;
status = pthread_attr_setschedparam(&attrReshape, &paramReshape);
if (status != 0) {
	perror("Unable to set params to attrReshape.\n");
}
paramH.sched_priority = 99;
status = pthread_attr_setschedparam(&attrH, &paramH);
if (status != 0) {
	perror("Unable to set params to attrH.\n");
}
paramProcessing.sched_priority = 99;
status = pthread_attr_setschedparam(&attrProcessing, &paramProcessing);
if (status != 0) {
	perror("Unable to set params to attrProcessing.\n");
}
paramS.sched_priority = 99;
status = pthread_attr_setschedparam(&attrS, &paramS);
if (status != 0) {
	perror("Unable to set params to attrS.\n");
}
paramFind.sched_priority = 99;
status = pthread_attr_setschedparam(&attrFind, &paramFind);
if (status != 0) {
	perror("Unable to set params to attrFind.\n");
}
paramShow.sched_priority = 99;
status = pthread_attr_setschedparam(&attrShow, &paramShow);
if (status != 0) {
	perror("Unable to set params to attrShow.\n");
}
////////create

pthread_create(&thRead, &attrRead, (void *) &readFrame, NULL);
pthread_join(thRead, NULL);

pthread_create(&thPredict, &attrPredict, (void *) &prediction, NULL);
pthread_join(thPredict, NULL);

pthread_create(&thCrop, &attrCrop, (void *) &crop, NULL);
pthread_join(thCrop, NULL);

pthread_create(&thR, &attrR, (void *) &R, NULL);
pthread_join(thR, NULL);

pthread_create(&thPn, &attrPn, (void *) &Pn, NULL);
pthread_join(thPn, NULL);

pthread_create(&thRotation, &attrRotation, (void *) &rotation, NULL);
pthread_join(thRotation, NULL);

pthread_create(&thReshape, &attrReshape, (void *) &Reshape, NULL);
pthread_join(thReshape, NULL);

pthread_create(&thH, &attrH, (void *) &Hfilter, NULL);
pthread_join(thH, NULL);

pthread_create(&thProcessing, &attrProcessing, (void *) &preprocessing, NULL);
pthread_join(thProcessing, NULL);

pthread_create(&thS, &attrS, (void *) &S, NULL);
pthread_join(thS, NULL);

pthread_create(&thFind, &attrFind, (void *) &Find, NULL);
pthread_join(thFind, NULL);

pthread_create(&thShow, &attrShow, (void *) &showFrame, NULL);
pthread_join(thShow, NULL);
//////////////////////////////////////////////////////////////////////
void * prediction(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in prediction.. \n");
			exit(-1);
		}
////////////////////////////////////////////////
		if (frame_counter>4)
        {
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
            ex[frame_counter]=xcenter;
            ey[frame_counter]=ycenter;
            Pfx1=xcenter;
            Pfy1=ycenter;
        }
////////////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * readFrame(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in readFrame.. \n");
			exit(-1);
		}
////////////////////////////////////////////////
        cvReleaseImage(&prev_frame);
        prev_frame=cvCloneImage(cf_gray);
        curr_frame = cvQueryFrame( capture );
        if(!curr_frame)
            break;
        cvCvtColor(curr_frame,cf_gray,CV_BGR2GRAY);
////////////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * crop(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in crop.. \n");
			exit(-1);
		}
///////////////////////////////////////
        if (frame_counter>start_frame)
        {
normal_crop(xcenter,ycenter,frame_szx,frame_szy,half_lines,half_cols,cf_gray,prev_frame,frag_lines,frag_cols,
frag_lines,frag_cols,Pfx1,Pfy1,curr_selection,crop,&xadp,&yadp,&out_of_bounds);
        }
///////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * ThreadR(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in ThreadR.. \n");
			exit(-1);
		}
//////////////////////////////////////////
		cvReleaseMat(&M);
		M=cvCloneMat(cvGetMat(curr_selection,cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
		mat2grayM(M,frag_lines,frag_cols);
		matrix_multiplication(gauss_win,M,cSelection,frag_lines,frag_cols);
		sig=in_noise_est(cSelection,frag_lines,frag_cols);
		est_rho(&cSelection->data.fl,frag_lines,frag_cols,&rho_x);
		rho_x=rho_x-(35*sig);
		rho_x=weighing_filters*rho_x+(1-weighing_filters)*rho_old;
		rho_old=rho_x;
		rho_y=rho_x;
		rho_x=0.2;
		vari=var(&cSelection->data.fl,frag_lines,frag_cols);
		for (int i = 0; i < frag_lines; ++i)
		{
		    for (int j = 0; j < frag_cols; ++j)
		    {
		        pot1=((ks[i][j]-half_cols)/frag_lines);
		        pot2=((ls[i][j]-half_lines)/frag_cols);
		        R[i][j]=vari*pow(rho_x,cabs(pot1))*pow(rho_y,cabs(pot2));
		    }
		}
////////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * ThreadPn(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in ThreadPn.. \n");
			exit(-1);
		}
//////////////////////////////////////
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
//////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * rotation(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in rotation.. \n");
			exit(-1);
		}
/////////////////////////////////////////
        cvReleaseMat(&imgsTrue[0]);
        cvReleaseMat(&q[0]);
        Ipltmp1=cvCloneImage(cvGetImage(M,&stub1));
        Ipltmp2=cvCloneImage(cvGetImage(cSelection,&stub2));
        imgsTrue[0]=cvCloneMat(M);
        q[0]=cvCloneMat(cSelection);
        for (int i = 0; i < 4; ++i)
        {
        	cvReleaseMat(&imgsTrue[i+1]);
        	cvReleaseMat(&imgsTrue[i+5]);
        	cvReleaseMat(&q[i+1]);
        	cvReleaseMat(&q[i+5]);
            imgsTrue[i+1]=cvCloneMat(cvGetMat(rotateImage(Ipltmp1,3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
            imgsTrue[i+5]=cvCloneMat(cvGetMat(rotateImage(Ipltmp1,-3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
            q[i+1]=cvCloneMat(cvGetMat(rotateImage(Ipltmp2,3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
            q[i+5]=cvCloneMat(cvGetMat(rotateImage(Ipltmp2,-3*(i+1)),cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
        }
        cvReleaseImage(&Ipltmp1);
        cvReleaseImage(&Ipltmp2);
/////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * Reshape(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in Reshape.. \n");
			exit(-1);
		}
/////////////////////////////////////////
		for (int i = 0; i < 9; ++i)
		{
		    srfft2(imgsTrue[i],fftmp,frag_lines,frag_cols);
		    creshape(frag_lines,frag_cols,fftmp,T,i);

		    srfft2(q[i],fftmp,frag_lines,frag_cols);
		    creshape(frag_lines,frag_cols,fftmp,Q,i);
		}
/////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * Hfilter(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in Hfilter.. \n");
			exit(-1);
		}
/////////////////////////////////////////
        t1matXmat(Q,T,sdTmpMat,lenc,9,lenc,9);
        Cinverse(sdTmpMat,sdTmpMati,9);
        matXmat(T,sdTmpMati,Q,lenc,9,9,9);
        matXvec(Q,c,sdfilter,lenc,9,9,1);
        creshapeB(frag_lines,frag_cols, sdfilter,fftmp2);
        if (frame_counter>start_frame)
        {
            for (int i = 0; i < frag_lines; ++i)
            {
                for (int j = 0; j < frag_cols; ++j)
                {
                    fftmp2[i][j]=(weighing_filters*fftmp2[i][j])+((1-weighing_filters)*Hold[i][j]);
                }
            }
        }
        matCpy(frag_lines,frag_cols,fftmp2,Hold,1);
/////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * preprocessing(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in preprocessing.. \n");
			exit(-1);
		}
/////////////////////////////////////////
        cvReleaseMat(&scene);
        scene=cvCloneMat(cvGetMat(crop,cvCreateMat(img32->height,img32->width,CV_32FC1),0,0));
        
        matrix_multiplication(gauss_win,scene,scene,frag_lines,frag_cols);
        med=avg(&scene->data.fl,frag_lines,frag_cols);
        subsMat(frag_lines,frag_cols,scene,scene,med);

        ifft2(fftmp2,filter,frag_lines,frag_cols);
        subsMatrix(frag_lines,frag_cols,filter,filter,cMatMean(frag_lines,frag_cols,filter));
        
        srfft2(scene,F,frag_lines,frag_cols);
        fftshift(&F,frag_lines,frag_cols);

        fft2(filter,fftmp2,frag_lines,frag_cols);//fftmp2=H ///filter?
        fftshift(&fftmp2,frag_lines,frag_cols);
        
        cconj(fftmp2,conjH,frag_lines,frag_cols);
        cconj(F,conjF,frag_lines,frag_cols);
        cmatrix_multiplication(F,conjF,multmp,frag_lines,frag_cols);
        ifft2(multmp,den1,frag_lines,frag_cols);
        ifftshift(&den1,frag_lines,frag_cols);
        cmatrix_multiplication(fftmp2,conjH,multmp,frag_lines,frag_cols);
        ifft2(multmp,den2,frag_lines,frag_cols);
        ifftshift(&den2,frag_lines,frag_cols);
    ///////////fin separacion
        cmatrix_multiplication(F,conjH,NUM,frag_lines,frag_cols);
        ifft2(NUM,num,frag_lines,frag_cols);
        ifftshift(&num,frag_lines,frag_cols);
/////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * ThreadS(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in ThreadS.. \n");
			exit(-1);
		}
/////////////////////////////////////////
        for (int i = 0; i < frag_lines; ++i)
        {
            for (int j = 0; j < frag_cols; ++j)
            {
                stmp=cabs(creal(num[i][j]/(0.1+den1[i][j]*den2[i][j])));
                s[i][j]=stmp*stmp;
            }
        }
/////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * Find(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in Find.. \n");
			exit(-1);
		}
/////////////////////////////////////////
        DC=calcDCfast(s,frag_lines,frag_cols);
        printf("%f\n",DC);
        if (DC>0.4)
        {
            findMax(s,frag_lines,frag_cols,&xcenter,&ycenter);
            xcenter=Pfx1-1+yadp+(xcenter-half_lines);//yadp
            ycenter=Pfy1-1+xadp+(ycenter-half_cols);//xadp
            tmpf=1;
        }
/////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}
void * Show(void *args) {
	struct timespec periodActivation, nextActivation, now;
	int period_time;
	period_time=400000000;
	periodActivation.tv_sec = 0;
	periodActivation.tv_nsec = period_time;
	nextActivation = initialTime;
	while (1){
		clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &nextActivation, NULL);
		clock_gettime(CLOCK_MONOTONIC, &now);
		if (timespec_cmp(&now, &nextActivation) > 0) {//validar tiempo de ejecucion
			fprintf(stderr, "Job activation lost in Show.. \n");
			exit(-1);
		}
/////////////////////////////////////////
        pearson=0.4;
        if (pearson>=0.4)
        {
            if (out_of_bounds==1)
            {
                if (DC>0.4)
                {
                    cvRectangle(curr_frame,cvPoint(ycenter-(half_cols/2)-1,xcenter-(half_lines/2)-1),cvPoint(ycenter+(half_cols/2)-1,xcenter+(half_lines/2-1)),cvScalar(255,40,40,0),3,8,0);
                    cvShowImage("Motion", curr_frame );
                }
                else
                {
                    cvRectangle(curr_frame,cvPoint(ycenter-(half_cols/2)-1,xcenter-(half_lines/2)-1),cvPoint(ycenter+(half_cols/2)-1,xcenter+(half_lines/2)-1),cvScalar(40,255,40,0),3,8,0);
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
/////////////////////////////////////////
		timespec_add(&nextActivation, &periodActivation);
	}
	pthread_exit(NULL);
}