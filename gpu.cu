#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <errno.h>


#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

#define MSG_LEN 10
#define HASH_LEN 30
#define THREADS_PER_BLOCK 512

// Round constants
__constant__ static const uint64_t RC[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a,
    0x8000000080008000, 0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008a,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 
    0x000000000000800a, 0x800000008000000a, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};


// Rotation offsets
__constant__ static const int r[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};



__constant__ static const int piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
};



// Updates the state with 24 rounds
__device__ void keccakf(uint64_t *state){
    int i, j;
    uint64_t temp, C[5];

    for (int round = 0; round < 24; round++) {
        // Theta
        for (i = 0; i < 5; i++) {
            C[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
        }

        for (i = 0; i < 5; i++) {
            temp = C[(i + 4) % 5] ^ ROTL64(C[(i + 1) % 5], 1);
            for (j = 0; j < 25; j += 5) {
                state[j + i] ^= temp;
            }
        }

        // Rho Pi
        temp = state[1];
        for (i = 0; i < 24; i++) {
            j = piln[i];
            C[0] = state[j];
            state[j] = ROTL64(temp, r[i]);
            temp = C[0];
        }

        //  Chi
        for (j = 0; j < 25; j += 5) {
            for (i = 0; i < 5; i++) {
                C[i] = state[j + i];
            }
            for (i = 0; i < 5; i++) {
                state[j + i] ^= (~C[(i + 1) % 5]) & C[(i + 2) % 5];
            }
        }

        //  Iota
        state[0] ^= RC[round];
    }
}



__global__ void keccak__offset(uint8_t *message_, unsigned long numbytes){
	int message_len = MSG_LEN; 
	uint64_t state[25];    
    uint8_t temp[144];
    int rsize = 136;            // 200 - 2 * 32
    int rsize_byte = 17;        // rsize / 8
    
    uint8_t message[MSG_LEN];

    unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * MSG_LEN) + (threadIdx.x * MSG_LEN);
  	if (offset >= numbytes) {  return; }

  	memcpy(message, &message_[offset], MSG_LEN);

  	memset(state, 0, sizeof(state));

  	// for ( ; message_len >= rsize; message_len -= rsize, (uint8_t *)message += rsize) {
        for (int i = 0; i < rsize_byte; i++) {
            state[i] ^= ((uint64_t *) message)[i];
        }
        keccakf(state);
    // }

    // Calculating the last state block and padding the result
    memcpy(temp, message, message_len);
    temp[message_len++] = 1;
    memset(temp + message_len, 0, rsize - message_len);
    temp[rsize - 1] |= 0x80;

    for (int i = 0; i < rsize_byte; i++) {
        state[i] ^= ((uint64_t *) temp)[i];
    }

    keccakf(state);
    __syncthreads();
}


void hashdemo(uint8_t *message_, unsigned long numbytes){
  uint8_t *message;
  // uint8_t *ctx_key_d, *ctx_enckey_d;

// RC[24]
  cudaMemcpyToSymbol(RC, RC, sizeof(uint64_t)*24);
  cudaMemcpyToSymbol(r, r, sizeof(int)*24);
  cudaMemcpyToSymbol(piln, piln, sizeof(int)*24);



  cudaMalloc((void**)&message, numbytes);


  cudaMemcpy(message, message_, numbytes, cudaMemcpyHostToDevice);

  dim3 dimBlock(ceil((double)numbytes / (double)(THREADS_PER_BLOCK * MSG_LEN)));
  dim3 dimGrid(THREADS_PER_BLOCK);


  keccak__offset<<<dimBlock, dimGrid>>>(message, numbytes);


  cudaFree(message);
  // cudaFree(ctx_key_d);
  // cudaFree(ctx_enckey_d);
}

__global__ void GPU_init() { }


int main(){

  // open file
  FILE *file;
  uint8_t *buf; // file buffer
  unsigned long numbytes;
  char *fname;
  clock_t start, enc_time, dec_time, end;
  int mili_sec, i;
  int padding;
  //key: 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 14 15 16 17 18 19 1a 1b 1c 1d 1e 1f
    /* create a key vector */
  uint8_t key[32];

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess){
    printf("Error: %s\n", cudaGetErrorString(error_id));
    printf("Exiting...\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0){
    printf("There are no available device(s) that support CUDA\n");
    exit(EXIT_FAILURE);
  }


  // handle txt file
  fname = "input.txt";  
  file = fopen(fname, "r");
  if (file == NULL) {printf("File %s doesn't exist\n", fname); exit(1); }
  printf("Opened file %s\n", fname);
  fseek(file, 0L, SEEK_END);
  numbytes = ftell(file);
  printf("Size is %lu\n", numbytes);

  // copy file into memory
  fseek(file, 0L, SEEK_SET);
  buf = (uint8_t*)calloc(numbytes, sizeof(uint8_t));
  if(buf == NULL) exit(1);
  if (fread(buf, 1, numbytes, file) != numbytes)
  {
    printf("Unable to read all bytes from file %s\n", fname);
    exit(EXIT_FAILURE);
  }
  fclose(file);


  // this is to force nvcc to put the gpu initialization here
  GPU_init<<<1, 1>>>();

  // encryption
  start = clock();
  hashdemo(buf, numbytes);
  end = clock();
  printf("time used:%f\n",  (double)(end - start) / CLOCKS_PER_SEC);
  printf("CPU encryption throughput: %f bytes/second\n",  (double)(numbytes) / ((double)(end - start) / CLOCKS_PER_SEC));


  free(buf);
  return EXIT_SUCCESS;
}