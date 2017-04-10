#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cassert>


#define check_cuda_call(ans) { _check((ans), __FILE__, __LINE__); }
inline void _check(cudaError_t code, char *file, int line)
{
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}


class EventTimer {
public:
  EventTimer() : mStarted(false), mStopped(false) {
    cudaEventCreate(&mStart);
    cudaEventCreate(&mStop);
  }
  ~EventTimer() {
    cudaEventDestroy(mStart);
    cudaEventDestroy(mStop);
  }
  void start(cudaStream_t s = 0) {
    cudaEventRecord(mStart, s); 
    mStarted = true;
    mStopped = false;
  }
  void stop(cudaStream_t s = 0)  {
    assert(mStarted);
    cudaEventRecord(mStop, s); 
    mStarted = false;
    mStopped = true;
  }
  float elapsed() {
    assert(mStopped);
    if (!mStopped) return 0; 
    cudaEventSynchronize(mStop);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, mStart, mStop);
    return elapsed;
  }

private:
  bool mStarted, mStopped;
  cudaEvent_t mStart, mStop;
};


__global__ void state_setup(curandStateXORWOW_t* states, int w, int h)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= w || y >= h) {
    return;
  }
  int i = x + y * w;
  curand_init(clock64(), x, 0, states + i);
}


__global__ void write(int *buf, int w, int h, float k, curandStateXORWOW_t* states)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x >= w || y >= h) {
    return;
  }
  int i = x + y * w;
  buf[i] = curand(states + i) & 1 ? 0x00000000 : 0xffffffff;
  // (x << 24 | y << 16 | x << 8 | y) * k;
}


curandStateXORWOW_t* states;

void state_setup(int w, int h)
{
  check_cuda_call(cudaMalloc(&states, w * h * sizeof(curandStateXORWOW_t)));
  check_cuda_call(cudaMemset(states, 0, w * h * sizeof(curandStateXORWOW_t)));
  dim3 dim_block(32, 16); // 32 * 16 = 512;
  dim3 dim_grid(((w + dim_block.x - 1) / dim_block.x),
                 (h + dim_block.y - 1) / dim_block.y);
  state_setup<<<dim_grid, dim_block>>>(states, w, h);
}


void cuda_write(int* buf, int w, int h, float k)
{
  dim3 dim_block(32, 16); // 32 * 16 = 512;
  dim3 dim_grid(((w + dim_block.x - 1) / dim_block.x),
                 (h + dim_block.y - 1) / dim_block.y);
  EventTimer t;
  t.start();
  write<<<dim_grid, dim_block>>>(buf, w, h, k, states);
  t.stop();
  //printf("kernel time: %f\n", t.elapsed());
}


void state_destroy()
{
  check_cuda_call(cudaFree(states));
}








