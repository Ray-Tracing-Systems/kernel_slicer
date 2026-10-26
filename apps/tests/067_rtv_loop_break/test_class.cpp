#include "test_class.h"

void LoopBreak::CollatzBlock(uint32_t tid, uint32_t a_maxIters, const uint32_t* a_start, uint32_t* a_steps, uint32_t* a_peak, uint32_t a_passesNum)
{
  for(uint32_t i = 0; i < tid; i++)
    Collatz(i, a_maxIters, a_start, a_steps, a_peak);
}

void LoopBreak::Collatz(uint32_t tid, uint32_t a_maxIters, const uint32_t* a_start, uint32_t* a_steps, uint32_t* a_peak)
{
  uint32_t value, count, peak;
  kernel_Init(tid, a_start, &value, &count, &peak);

  for(uint32_t iter = 0; iter < a_maxIters; iter++)
  {
    if(!kernel_Step(tid, &value))
      break;
    kernel_Accumulate(tid, &value, &count, &peak);
  }

  kernel_Write(tid, &count, &peak, a_steps, a_peak);
}

void LoopBreak::kernel_Init(uint32_t tid, const uint32_t* a_start, uint32_t* a_value, uint32_t* a_count, uint32_t* a_currPeak)
{
  *a_value = a_start[tid];
  *a_count = 0;
  *a_currPeak = a_start[tid];
}

bool LoopBreak::kernel_Step(uint32_t tid, uint32_t* a_value)
{
  const uint32_t value = *a_value;
  if(value <= 1)
    return false;
  *a_value = (value % 2 == 0) ? value/2 : 3*value + 1;
  return true;
}

void LoopBreak::kernel_Accumulate(uint32_t tid, const uint32_t* a_value, uint32_t* a_count, uint32_t* a_currPeak)
{
  *a_count = *a_count + 1;
  if(*a_value > *a_currPeak)
    *a_currPeak = *a_value;
}

void LoopBreak::kernel_Write(uint32_t tid, const uint32_t* a_count, const uint32_t* a_currPeak, uint32_t* a_outSteps, uint32_t* a_outPeak)
{
  a_outSteps[tid] = *a_count;
  a_outPeak [tid] = *a_currPeak;
}
