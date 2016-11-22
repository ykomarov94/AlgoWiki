#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define nDims 10
#define nC 11
#define frand() (((float) rand())/RAND_MAX)
#define rootProcess 0

#define _TAGNUMVECS 1
#define _TAGVECS 2
#define _TAGRESULTS 4
#define _TAGCOUNT 8
#define _TAGJ 16
#define _TAGU 32
#define _TAGCONT 64

int _zeros(char *src, int m, int n) {
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*(src + i*n + j) = 0;
		}
	}
	return 0;
}

int _zeroC(float *src, int m, int n){
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*(src + i*n + j) = 0;
		}
	}
	return 0;
}

int _zeroCopy(float* dst, float *src, int m, int n) {
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*(dst + i*n + j) = *(src +i*n + j);
			*(src +i*n + j) = 0.0;
		}
	}
	return 0;
}

float __sq(float src) {
	return src*src;
}

float dSq(float* a, float* b) {
	int i;
	float d = 0.0, tmp;
	for (i = 0; i < nDims; i++) {
		tmp = *(a+i) - *(b+i);
		d += __sq(tmp);
	}
	return d;
}

int _initU(float* u, int m, int n) {
	int i, j;
	srand(time(NULL));
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*(u +i*n + j) = frand();
		}
	}
	return 0;
}

int _addVec(float* to, float* from) {
	int i;
	for (i = 0; i < nDims; i++) {
		*(to+i) += *(from+i);
	}
	return 0;
}

int _initCNew(float* cNew, float* u) {
	int i,j;
	for (i = 0; i < nC; i++){
		for (j = 0; j < nDims; j++) {
			*(cNew + i*nDims + j) = *(u + i*nDims + j);
		}
	}
	return 0;
}

int main(int argc, char **argv){

	MPI_Status status;
	int myId, nThreads, ierr, iThread;
	
	char *ptr;
	int counter = 0;
	double timeTrack;
	int i, j;
	const unsigned long int nVecs = roundf(strtof(argv[1], &ptr));
	float *u;
	char *M;
	float *c;
	float *cNew;
	int jMin;
	float dij, dijMin;
	double jPrev = FLT_MAX;
	double jCur = FLT_MAX / 2;
	float eps = 1e-2;
	int cNewCount[nC];
	const int cSize = sizeof(float)*nC*nDims;
	const int mSize = sizeof(char)*nVecs*nC;
	
	float *cTmp;
	int cCountTmp[nC];
	double jTmp;
	int uPerThread;
	unsigned int isNotFinished;
//----------------------------------------- Parallel code starts here
	ierr = MPI_Init(&argc, &argv);

	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &nThreads);

	if (myId == rootProcess) {
	//------------------------------------------------- Master Source
		
//M = (char*) malloc(mSize);
		c = (float*) malloc(cSize);
		cNew = (float*) malloc(cSize);
		cTmp = (float*) malloc(cSize);
		u = (float*) malloc(sizeof(float)*nVecs*nDims);
		uPerThread = truncf(nVecs/(nThreads-2));


		_initU(u,nVecs,nDims);
		_initCNew(cNew, u);

		timeTrack = clock();


		//send u for each process
		for (iThread = 1; iThread < nThreads-1; iThread++){
			ierr = MPI_Send( (void*) (u + (iThread-1)*nDims*uPerThread), nDims*uPerThread, MPI_FLOAT,
				iThread, _TAGU, MPI_COMM_WORLD);
		}

		ierr = MPI_Send( (void*) (u+(nThreads-1)*nDims*uPerThread), nDims*(nVecs - (nThreads-2)*uPerThread),
			MPI_FLOAT, nThreads - 1, _TAGU, MPI_COMM_WORLD);


		isNotFinished = 1;

		while (abs(jPrev-jCur) >= eps) {
			counter++;
			//_zeros(M,nVecs,nC);
			_zeroCopy(c, cNew, nC, nDims);

			jPrev = jCur;
			jCur = 0;
			for (i = 0; i < nC; i++) {
				cNewCount[i] = 0;
			}

			ierr = MPI_Send( &isNotFinished, 1, MPI_INT,
				rootProcess+1, _TAGCONT, MPI_COMM_WORLD);

			for (iThread = 1; iThread < nThreads; iThread++) {
				// send data to slaves
				ierr = MPI_Send( (void*)c, nC*nDims, MPI_FLOAT,
					iThread, _TAGVECS, MPI_COMM_WORLD);
			}

			for (iThread = 1; iThread < nThreads; iThread++) {
				// receive and maintain data
				ierr = MPI_Recv( (void*)cTmp, nC*nDims, MPI_FLOAT,
					iThread, _TAGRESULTS, MPI_COMM_WORLD, &status);
				ierr = MPI_Recv( &cCountTmp, nC, MPI_INT,
					iThread, _TAGCOUNT, MPI_COMM_WORLD, &status);
				ierr = MPI_Recv( &jTmp, 1, MPI_DOUBLE,
					iThread, _TAGJ, MPI_COMM_WORLD, &status);

				for (i = 0; i < nC; i++){
					_addVec(cNew + i*nDims, cTmp + i*nDims);
					cNewCount[i] += cCountTmp[i];
					
				}
				jCur += jTmp;
			}

			for (i = 0; i < nC; i++){
				for (j = 0; j < nDims; j++) {
					*(cNew + i*nDims + j) /= cNewCount[i];
				}
			}
		}

		isNotFinished = 0;
		ierr = MPI_Send( &isNotFinished, 1, MPI_INT,
			rootProcess+1, _TAGCONT, MPI_COMM_WORLD);

timeTrack = (clock() - timeTrack)/CLOCKS_PER_SEC;
//printf("finished! %f by %d iterations within %.4lf seconds\n", jCur, counter, timeTrack);

		FILE *outFile = fopen("results.csv", "a");
		if (outFile != NULL){
			fprintf(outFile,"%d,%ld,%.4lf,%d\n",nThreads, nVecs, timeTrack,counter);
			fclose(outFile);
		}


	//free(M);
	free(c);
	free(cNew);
	free(cTmp);
	free(u);
	}
	else{
	// Slave source ------------------------------------------------
		uPerThread = truncf(nVecs/(nThreads-2));
		if (myId == (nThreads-1)){
			uPerThread = nVecs - (nThreads-2)*uPerThread;
		}

		c = (float*) malloc(cSize);
		cNew = (float*) malloc(cSize);
		u = (float*) malloc(sizeof(float)*nDims*uPerThread);

		ierr = MPI_Recv( (void*) u, nDims*uPerThread, MPI_FLOAT,
			rootProcess, _TAGU, MPI_COMM_WORLD, &status);

		ierr = MPI_Recv( &isNotFinished, 1, MPI_INT,
				myId-1, _TAGCONT, MPI_COMM_WORLD, &status);
		if (myId < nThreads-1){
			ierr = MPI_Send( &isNotFinished, 1, MPI_INT,
				myId+1, _TAGCONT, MPI_COMM_WORLD);
		}
		

		while (isNotFinished) {
			ierr = MPI_Recv( (void*) c, nC*nDims, MPI_FLOAT,
				rootProcess, _TAGVECS, MPI_COMM_WORLD, &status);
			

			_zeroC(cNew, nC, nDims);
			jCur = 0;
			for (i = 0; i < nC; i++) {
				cNewCount[i] = 0;
			}

			for (i = 0; i < uPerThread; i++) {
			dijMin = FLT_MAX;

				for (j = 0; j < nC; j++) {
					dij = dSq(u+i*nDims,c+j*nDims);
					if (dij < dijMin) {
						dijMin = dij;
						jMin = j;
					}
				}
				//*(M + i*nC + j) = 1;
				_addVec(cNew+jMin*nDims, u+i*nDims);
				cNewCount[jMin]++;
				jCur += dijMin;
			}

			ierr = MPI_Send( cNew, nC*nDims, MPI_FLOAT,
				rootProcess, _TAGRESULTS, MPI_COMM_WORLD);
			ierr = MPI_Send( &cNewCount, nC, MPI_INT,
				rootProcess, _TAGCOUNT, MPI_COMM_WORLD);
			ierr = MPI_Send( &jCur, 1, MPI_DOUBLE,
				rootProcess, _TAGJ, MPI_COMM_WORLD);

			ierr = MPI_Recv( &isNotFinished, 1, MPI_INT,
				myId-1, _TAGCONT, MPI_COMM_WORLD, &status);
			if (myId < nThreads-1){
				ierr = MPI_Send( &isNotFinished, 1, MPI_INT,
					myId+1, _TAGCONT, MPI_COMM_WORLD);
			}
		}
		free(c);
		free(cNew);
		free(u);

	}
	ierr = MPI_Finalize();
}