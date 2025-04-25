
#include "LiteMath.h"
#include <extended/lm_device_vector.h> // also from LiteMath
#include "test_class.h"
#include <cfloat>
#include <mutex>

template<typename T> inline size_t ReduceAddInit(LiteMathExtended::device_vector<T>& a_vec, size_t a_targetSize, size_t a_threadsNum) 
{ 
  const size_t blockSize = 256;
  size_t currOffset = 0;
  size_t currSize   = a_threadsNum/blockSize;
  while (currSize > 1) 
  {
    currOffset += currSize*a_targetSize;
    size_t numBlocks = (currSize + blockSize - 1) / blockSize;
    //BlockReduce (...)
    currSize = numBlocks;
  }

  currOffset += a_targetSize;

  a_vec.reserve(a_targetSize + currOffset);
  a_vec.resize(a_targetSize);
  cudaMemset(a_vec.data(), 0, a_vec.capacity()*sizeof(T)); 

  return (a_threadsNum / blockSize); 
}

template<typename T>
__global__ void BlockReduce(const T* in_data, T* out_data, uint32_t a_threadsNum, uint32_t a_vecSize)
{
  const uint32_t tid1 = blockIdx.x * blockDim.x + threadIdx.x;
  //const uint32_t tid2 = blockIdx.x/a_vecSize * blockDim.x + threadIdx.x;

  __shared__ T sdata[256*1*1]; 
  if(tid1 < a_threadsNum)
    sdata[threadIdx.x] = in_data[tid1];
  else
    sdata[threadIdx.x] = 0;
  __syncthreads();

  if (threadIdx.x < 128)
    sdata[threadIdx.x] += sdata[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x < 64)
    sdata[threadIdx.x] += sdata[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x < 32) sdata[threadIdx.x] += sdata[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x < 8)  sdata[threadIdx.x] += sdata[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x < 4)  sdata[threadIdx.x] += sdata[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x < 2)  sdata[threadIdx.x] += sdata[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x < 1)  sdata[threadIdx.x] += sdata[threadIdx.x + 1];
  __syncthreads();

  if(threadIdx.x == 0)
  {
    out_data[blockIdx.x] = sdata[0];
  }
}

template<typename T> inline void ReduceAddComplete(LiteMathExtended::device_vector<T>& a_vec, size_t a_threadsNum) 
{ 
  const size_t blockSize = 256;
  size_t currSize   = a_threadsNum/blockSize; 
  
  {
    //std::vector<T> debug(currSize);
    //cudaMemcpy(debug.data(), a_vec.data() + a_vec.size(), debug.size()*sizeof(T), cudaMemcpyDeviceToHost);
    std::vector<T> debug(a_vec.size());
    cudaMemcpy(debug.data(), a_vec.data(), debug.size()*sizeof(T), cudaMemcpyDeviceToHost);
    int a = 2;
  }

  size_t currOffset = currSize; //*a_vec.size();
  size_t oldOffset  = a_vec.size();
  while (currSize > 1) 
  {
    size_t numBlocks       = (currSize + blockSize - 1) / blockSize;
    size_t outOffset       = (numBlocks == 1) ? 0 : oldOffset + currOffset;
    //size_t numBlocksActual = numBlocks*a_vec.size();
    BlockReduce <T> <<<numBlocks, blockSize>>> (a_vec.data() + oldOffset, a_vec.data() + outOffset, currSize, uint32_t(a_vec.size()));
    currSize  = numBlocks;
    oldOffset = currOffset;
    currOffset += currSize*a_vec.size();
  }
}

namespace SimpleTest_Generated_DEV
{
  using _Bool = bool;

  //template<typename T, typename IndexType> // TODO: pass block size via template parameter
  //__device__ inline void ReduceAdd(LiteMathExtended::device_vector<T>& a_vec, IndexType offset, T val)
  //{
  //  //__shared__ T sval;
  //  //if(threadIdx.x == 0)
  //  //  sval = 0;
  //  //__syncthreads();
  //  //atomicAdd(&sval, val);
  //  //__syncthreads();
  //  //if(threadIdx.x == 0)
  //  //  atomicAdd(a_vec.data() + offset, sval);
  //
  //  __shared__ T sdata[256*1*1]; 
  //  sdata[threadIdx.x] = val;
  //  __syncthreads();
  //  if (threadIdx.x < 128)
  //    sdata[threadIdx.x] += sdata[threadIdx.x + 128];
  //  __syncthreads();
  //  if (threadIdx.x < 64)
  //    sdata[threadIdx.x] += sdata[threadIdx.x + 64];
  //  __syncthreads();
  //  if (threadIdx.x < 32) sdata[threadIdx.x] += sdata[threadIdx.x + 32];
  //  __syncthreads();
  //  if (threadIdx.x < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
  //  __syncthreads();
  //  if (threadIdx.x < 8)  sdata[threadIdx.x] += sdata[threadIdx.x + 8];
  //  __syncthreads();
  //  if (threadIdx.x < 4)  sdata[threadIdx.x] += sdata[threadIdx.x + 4];
  //  __syncthreads();
  //  if (threadIdx.x < 2)  sdata[threadIdx.x] += sdata[threadIdx.x + 2];
  //  __syncthreads();
  //  if (threadIdx.x < 1)  sdata[threadIdx.x] += sdata[threadIdx.x + 1];
  //  __syncthreads();
  //
  //  if(threadIdx.x == 0)
  //  {
  //    atomicAdd(a_vec.data() + offset,  sdata[0]);
  //    //(a_vec.data() + a_vec.size())[blockIdx.x] = sdata[0];
  //  }
  //}

  template<typename T, typename IndexType> // TODO: pass block size via template parameter
  __device__ inline void ReduceAdd(LiteMathExtended::device_vector<T>& a_vec, IndexType offset, IndexType a_sizeAligned, T val)
  {
    //__shared__ T sval;
    //if(threadIdx.x == 0)
    //  sval = 0;
    //__syncthreads();
    //atomicAdd(&sval, val);
    //__syncthreads();
    //if(threadIdx.x == 0)
    //  atomicAdd(a_vec.data() + offset, sval);

    //__shared__ T sdata[256*1*1]; 
    //sdata[threadIdx.x] = val;
    //__syncthreads();
    //if (threadIdx.x < 128)
    //  sdata[threadIdx.x] += sdata[threadIdx.x + 128];
    //__syncthreads();
    //if (threadIdx.x < 64)
    //  sdata[threadIdx.x] += sdata[threadIdx.x + 64];
    //__syncthreads();
    //if (threadIdx.x < 32) sdata[threadIdx.x] += sdata[threadIdx.x + 32];
    //__syncthreads();
    //if (threadIdx.x < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16];
    //__syncthreads();
    //if (threadIdx.x < 8)  sdata[threadIdx.x] += sdata[threadIdx.x + 8];
    //__syncthreads();
    //if (threadIdx.x < 4)  sdata[threadIdx.x] += sdata[threadIdx.x + 4];
    //__syncthreads();
    //if (threadIdx.x < 2)  sdata[threadIdx.x] += sdata[threadIdx.x + 2];
    //__syncthreads();
    //if (threadIdx.x < 1)  sdata[threadIdx.x] += sdata[threadIdx.x + 1];
    //__syncthreads();

    if(threadIdx.x == 0)
    {
      //(a_vec.data() + a_vec.size())[blockIdx.x] += 1.0f; //sdata[0]; // a_sizeAligned*offset + 
      a_vec[blockIdx.x] = 1.0f; //sdata[0]; // a_sizeAligned*offset + 
    }
  }


  __device__ LiteMathExtended::device_vector<float> m_accum;
  struct UniformBufferObjectData
  {
  };
  __device__ UniformBufferObjectData ubo;
 
  __global__ void kernel1D_CalcAndAccum(const float* __restrict__  in_data, uint32_t a_threadsNum, float* __restrict__  a_out, uint32_t a_alignedSize)
  {
    //const int i = blockIdx.x * blockDim.x + threadIdx.x
  
    ReduceAdd<float, uint32_t>(m_accum, 0, a_alignedSize, 0.1f);
    //ReduceAdd<float, uint32_t>(m_accum, 1, a_alignedSize, 0.5f);
    //ReduceAdd<float, uint32_t>(m_accum, 2, a_alignedSize, 1.0f);
    //ReduceAdd<float, uint32_t>(m_accum, 3, a_alignedSize, 2.0f);
    //ReduceAdd<float, uint32_t>(m_accum, 4, a_alignedSize, 3.0f);
  }

  __global__ void kernel1D_CopyData(float* __restrict__  a_out, const float* __restrict__  a_in, uint32_t a_size)
  {
    const uint _threadID[3] = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z
    };
    const uint32_t i = uint32_t(_threadID[0]); 
    bool runThisThread = true;
    if(i >= a_size + 0)
      runThisThread = false;
    if(runThisThread) 
    {
    a_out[i] = a_in[i];
    }
  }

};

#include <memory>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <vector>
#include <string>
#include "test_class.h"

//#include <thrust/device_vector.h> // if use real thrust
//using thrust::device_vector;      // if use real thrust
using LiteMathExtended::device_vector;

class SimpleTest_Generated : public SimpleTest
{
public:

  SimpleTest_Generated()
  {
  }
  
  virtual ~SimpleTest_Generated()
  {
    m_accum_dev.resize(0);
    m_accum_dev.shrink_to_fit(); 
    cudaFree(m_pUBO); m_pUBO = nullptr;
  }

  void CommitDeviceData() override;
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override;
  void CopyUBOToDevice();
  void CopyUBOFromDevice();
  void UpdateDeviceVectors();

  void kernel1D_CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out, uint32_t a_alignedSize);
  void kernel1D_CopyData(float* a_out, const float* a_in, uint32_t a_size) override;
  
  void CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out) override;
  virtual void CalcAndAccumGPU(const float* in_data, uint32_t a_threadsNum, float* a_out);

  virtual void UpdateObjectContext(bool a_updateVec = true);
  virtual void ReadObjectContext(bool a_updateVec = true);

protected:
  device_vector<float> m_accum_dev;
  float m_exTimeCalcAndAccum[4] = {0,0,0,0};

  SimpleTest_Generated_DEV::UniformBufferObjectData* m_pUBO = nullptr;
  static std::mutex m_mtx;
};

std::mutex SimpleTest_Generated::m_mtx;

class SimpleTest_GeneratedDEV : public SimpleTest_Generated
{
public:

  SimpleTest_GeneratedDEV() {}
  
  void CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out) override {
    CalcAndAccumGPU(in_data, a_threadsNum, a_out);
  }

protected:
};

std::shared_ptr<SimpleTest> CreateSimpleTest_Generated()
{
  auto pObj = std::make_shared<SimpleTest_Generated>();
  return pObj;
}
std::shared_ptr<SimpleTest> CreateSimpleTest_Generated_DEV()
{
  auto pObj = std::make_shared<SimpleTest_GeneratedDEV>();
  return pObj;
}

void SimpleTest_Generated::CopyUBOToDevice()
{
  if(m_pUBO == nullptr)
    cudaMalloc(&m_pUBO, sizeof(SimpleTest_Generated_DEV::UniformBufferObjectData));
  
  SimpleTest_Generated_DEV::UniformBufferObjectData ubo;
  cudaMemcpy(m_pUBO, &ubo, sizeof(ubo), cudaMemcpyHostToDevice);
}

void SimpleTest_Generated::CopyUBOFromDevice()
{
  SimpleTest_Generated_DEV::UniformBufferObjectData ubo;
  cudaMemcpy(&ubo, m_pUBO, sizeof(ubo), cudaMemcpyDeviceToHost);
  if(m_accum.size() != m_accum_dev.size())
    m_accum.resize(m_accum_dev.size());
}

void SimpleTest_Generated::UpdateDeviceVectors() 
{
  m_accum_dev.reserve(m_accum.capacity());
  m_accum_dev.assign(m_accum.begin(), m_accum.end());
}

void SimpleTest_Generated::CommitDeviceData()
{
  UpdateDeviceVectors();
  CopyUBOToDevice();
}

void SimpleTest_Generated::UpdateObjectContext(bool a_updateVec)
{
  cudaMemcpyToSymbol(SimpleTest_Generated_DEV::ubo, m_pUBO, sizeof(SimpleTest_Generated_DEV::UniformBufferObjectData), 0, cudaMemcpyDeviceToDevice);
  if(a_updateVec)
  {
    cudaMemcpyToSymbol(SimpleTest_Generated_DEV::m_accum, &m_accum_dev, sizeof(LiteMathExtended::device_vector<float>));
  }
}

void SimpleTest_Generated::ReadObjectContext(bool a_updateVec)
{
  cudaMemcpyFromSymbol(m_pUBO, SimpleTest_Generated_DEV::ubo, sizeof(SimpleTest_Generated_DEV::UniformBufferObjectData), 0, cudaMemcpyDeviceToDevice);
  if(a_updateVec)
  {
    cudaMemcpyFromSymbol(&m_accum_dev, SimpleTest_Generated_DEV::m_accum, sizeof(LiteMathExtended::device_vector<float>));
  }
}

void SimpleTest_Generated::kernel1D_CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out, uint32_t a_alignedSize)
{
  dim3 block(256, 1, 1);
  dim3 grid((a_threadsNum + block.x - 1) / block.x, (1 + block.y - 1) / block.y, (1 + block.z - 1) / block.z);
  SimpleTest_Generated_DEV::kernel1D_CalcAndAccum<<<grid, block>>>(in_data, a_threadsNum, a_out, a_alignedSize);
}

void SimpleTest_Generated::kernel1D_CopyData(float* a_out, const float* a_in, uint32_t a_size)
{
  dim3 block(256, 1, 1);
  dim3 grid((a_size + block.x - 1) / block.x, (1 + block.y - 1) / block.y, (1 + block.z - 1) / block.z);
  SimpleTest_Generated_DEV::kernel1D_CopyData<<<grid, block>>>(a_out, a_in, a_size);
}

void SimpleTest_Generated::CalcAndAccumGPU(const float* in_data, uint32_t a_threadsNum, float* a_out)
{
  std::lock_guard<std::mutex> lock(m_mtx); // lock for UpdateObjectContext/ReadObjectContext to be ussied for this object only
  
  size_t alignedSize = ReduceAddInit(m_accum_dev, m_accum_dev.size(), a_threadsNum);
  UpdateObjectContext();
  
  kernel1D_CalcAndAccum(in_data, a_threadsNum, a_out, uint32_t(alignedSize));
  ReduceAddComplete(m_accum_dev, a_threadsNum);
  kernel1D_CopyData(a_out, m_accum_dev.data(), uint32_t(m_accum.size()));

  ReadObjectContext();
}

void SimpleTest_Generated::CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out)
{
  const float * in_dataHost = in_data;
  float * a_outHost = a_out;
  
  cudaEvent_t _start, _stop;
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
  
  cudaEventRecord(_start);
  cudaMalloc(&in_data, a_threadsNum*sizeof(const float ));
  cudaMalloc(&a_out, 5*sizeof(float ));
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTimeCalcAndAccum[3], _start, _stop);
  
  cudaEventRecord(_start);
  cudaMemcpy((void*)in_data, in_dataHost, a_threadsNum*sizeof(const float ), cudaMemcpyHostToDevice);
  CopyUBOToDevice();
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTimeCalcAndAccum[1], _start, _stop);
  
  cudaEventRecord(_start);
  CalcAndAccumGPU(in_data, a_threadsNum, a_out);
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTimeCalcAndAccum[0], _start, _stop);
  
  cudaEventRecord(_start);
  CopyUBOFromDevice();
  cudaMemcpy(a_outHost, a_out, 5*sizeof(float ), cudaMemcpyDeviceToHost);
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&m_exTimeCalcAndAccum[2], _start, _stop);
  
  cudaEventRecord(_start);
  cudaFree((void*)in_data);
  cudaFree(a_out);
  cudaEventRecord(_stop);
  cudaEventSynchronize(_stop);
  float _timeForFree = 0.0f;
  cudaEventElapsedTime(&_timeForFree, _start, _stop);
  m_exTimeCalcAndAccum[3] += _timeForFree;
  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
}


void SimpleTest_Generated::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "CalcAndAccum" || std::string(a_funcName) == "CalcAndAccumBlock")
  {
    a_out[0] = m_exTimeCalcAndAccum[0];
    a_out[1] = m_exTimeCalcAndAccum[1];
    a_out[2] = m_exTimeCalcAndAccum[2];
    a_out[3] = m_exTimeCalcAndAccum[3];
  }
}

