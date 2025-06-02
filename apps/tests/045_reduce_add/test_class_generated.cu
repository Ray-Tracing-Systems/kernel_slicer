#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include "LiteMath.h"
#include <extended/lm_device_vector.h> // also from LiteMath

#include "test_class.h"
#include <vector>
#include <cfloat>
#include <mutex>

template<typename T> inline size_t ReduceAddInit(std::vector<T>& a_vec, size_t a_targetSize) { return a_vec.size(); }
template<typename T> inline void   ReduceAddComplete(std::vector<T>& a_vec) { }

namespace SimpleTest_Generated_DEV
{
  using _Bool = bool;
  template<typename T, typename IndexType, int BLOCK_SIZE = 256>
  __device__ inline void ReduceAdd(LiteMathExtended::device_vector<T>& a_vec, IndexType offset, T val)
  {
    if(!isfinite(val))
      val = 0;
        
    // Определяем временное хранилище для CUB
    __shared__ typename cub::BlockReduce<T, BLOCK_SIZE>::TempStorage temp_storage;
  
    // Выполняем редукцию по блоку
    T block_sum = cub::BlockReduce<T, BLOCK_SIZE>(temp_storage).Sum(val);
  
    // Первый поток атомарно добавляет результат в глобальную память
    if(threadIdx.x == 0)
      atomicAdd(a_vec.data() + offset, block_sum);
  }
  template<typename T, typename IndexType> // TODO: pass block size via template parameter
  __device__ inline void ReduceAdd(LiteMathExtended::device_vector<T>& a_vec, IndexType offset, IndexType a_sizeAligned, T val)  { ReduceAdd<T,IndexType>(a_vec, offset, val); }


  struct UniformBufferObjectData
  {
    LiteMathExtended::device_vector<float> m_accum;
  };
  __device__ UniformBufferObjectData ubo;


  __device__ float atomicMin(float* address, float val) 
  {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float current_val = __int_as_float(old);
        if (val >= current_val) 
          break;  // Если новое значение не меньше, выходим
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
  }

  __device__ float atomicMax(float* address, float val) 
  {
    int* addr_as_int = (int*)address;
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float current_val = __int_as_float(old);
        if (val <= current_val) 
          break;  // Если новое значение не больше, выходим
        old = atomicCAS(addr_as_int, expected, __float_as_int(val));
    } while (expected != old);
    return __int_as_float(old);
  }
 
  __global__ void kernel1D_CalcAndAccum(const float* __restrict__  in_data, uint32_t a_threadsNum, float* __restrict__  a_out)
  {
    const uint _threadID[3] = {
      blockIdx.x * blockDim.x + threadIdx.x,
      blockIdx.y * blockDim.y + threadIdx.y,
      blockIdx.z * blockDim.z + threadIdx.z
    };
    const int i = int(_threadID[0]); 
    bool runThisThread = true;
    if(i >= a_threadsNum + 0)
      runThisThread = false;
    
    //float x = in_data[i];
    ReduceAdd<float, uint32_t>(ubo.m_accum, 0, 1.0f);
    ReduceAdd<float, uint32_t>(ubo.m_accum, 1, 2.0f);
    ReduceAdd<float, uint32_t>(ubo.m_accum, 2, 3.0f);
    ReduceAdd<float, uint32_t>(ubo.m_accum, 3, 4.0f);
    ReduceAdd<float, uint32_t>(ubo.m_accum, 4, 5.0f);
  
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
    if(runThisThread) {
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

  void kernel1D_CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out) override;
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
  ubo.m_accum.m_data     = m_accum_dev.data();
  ubo.m_accum.m_size     = m_accum_dev.size();
  ubo.m_accum.m_capacity = m_accum_dev.capacity();
  cudaMemcpy(m_pUBO, &ubo, sizeof(ubo), cudaMemcpyHostToDevice);
}

void SimpleTest_Generated::CopyUBOFromDevice()
{
  SimpleTest_Generated_DEV::UniformBufferObjectData ubo;
  cudaMemcpy(&ubo, m_pUBO, sizeof(ubo), cudaMemcpyDeviceToHost);
  if(m_accum.size() != ubo.m_accum.size())
  {
    m_accum.resize(ubo.m_accum.size());
    m_accum.resize(ubo.m_accum.size());
  }
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
}

void SimpleTest_Generated::ReadObjectContext(bool a_updateVec)
{
  cudaMemcpyFromSymbol(m_pUBO, SimpleTest_Generated_DEV::ubo, sizeof(SimpleTest_Generated_DEV::UniformBufferObjectData), 0, cudaMemcpyDeviceToDevice);
}

void SimpleTest_Generated::kernel1D_CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out)
{
  dim3 block(256, 1, 1);
  dim3 grid((a_threadsNum + block.x - 1) / block.x, (1 + block.y - 1) / block.y, (1 + block.z - 1) / block.z);
  SimpleTest_Generated_DEV::kernel1D_CalcAndAccum<<<grid, block>>>(in_data, a_threadsNum, a_out);
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
  UpdateObjectContext();
    ReduceAddInit(m_accum, m_accum.size());
  kernel1D_CalcAndAccum(in_data, a_threadsNum, a_out);
  ReduceAddComplete(m_accum);
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

