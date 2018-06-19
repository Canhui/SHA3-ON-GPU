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


// Round constants
const uint64_t RC[24] = {
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
const int r[24] = {
    1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14, 
    27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44
};

const int piln[24] = {
    10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4, 
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1 
};



// Updates the state with 24 rounds
void keccakf(uint64_t *state){
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



// Compute the keccak hash from the message and store it message the output
void keccak(const uint8_t *message, int message_len, uint8_t *output, int output_len){
    uint64_t state[25];    
    uint8_t temp[144];
    int rsize = 136;			// 200 - 2 * 32
    int rsize_byte = 17;		// rsize / 8

	// resets the state to 0
    memset(state, 0, sizeof(state));

    for ( ; message_len >= rsize; message_len -= rsize, message += rsize) {
        for (int i = 0; i < rsize_byte; i++) {
            state[i] ^= ((uint64_t *) message)[i];
		}
        keccakf(state);
    }
    
    // Calculating the last state block and padding the result
    memcpy(temp, message, message_len);
    temp[message_len++] = 1;
    memset(temp + message_len, 0, rsize - message_len);
    temp[rsize - 1] |= 0x80;

    for (int i = 0; i < rsize_byte; i++) {
        state[i] ^= ((uint64_t *) temp)[i];
	}

    keccakf(state);
    // memcpy(output, state, output_len);
}





int main(){

  // open file
  FILE *file;
  uint8_t *buf;
  unsigned long numbytes;
  char *fname;
  clock_t start, enc_time, dec_time, end;
  int mili_sec, i;
  int padding;
  uint8_t key[32];
  uint8_t output[HASH_LEN];
  unsigned long offset = 20;

  // handle txt file
  fname = "input.txt";
  file = fopen(fname, "r");
  fseek(file, 0L, SEEK_END);
  numbytes = ftell(file);
  printf("Size is %lu\n", numbytes);

  fseek(file, 0L, SEEK_SET);
  buf = (uint8_t*)calloc(numbytes, sizeof(uint8_t));
  if(buf == NULL) exit(1);
  if (fread(buf, 1, numbytes, file) != numbytes){
    printf("Unable to read all bytes from file %s\n", fname);
    exit(EXIT_FAILURE);
  }
  fclose(file);


  // start encrypt in CPU
  start = clock();
  // hashdemo(buf, numbytes);
  for (int i=0; i< 400000; i++){
  	keccak(buf, numbytes, output, HASH_LEN);
  	// hashdemo(buf, numbytes, output, HASH_LEN, offset);	
  }
  
  // encryptdemo(key, buf, numbytes);
  end = clock();
  printf("time used:%f\n",  (double)(end - start) / CLOCKS_PER_SEC);
  printf("CPU encryption throughput: %f bytes/second\n",  (double)(numbytes) / ((double)(end - start) / CLOCKS_PER_SEC));


  free(buf);
  return EXIT_SUCCESS;
}