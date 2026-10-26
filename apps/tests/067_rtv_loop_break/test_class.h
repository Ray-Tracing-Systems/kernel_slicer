#pragma once
#include <vector>
#include <cstdint>

// RTV: a kernel returns bool and the control function leaves the loop with 'break';
// every thread leaves the loop at its own iteration (Collatz sequence)
class LoopBreak
{
public:
  LoopBreak(){}

  virtual void CollatzBlock(uint32_t tid, uint32_t a_maxIters, const uint32_t* a_start, uint32_t* a_steps, uint32_t* a_peak, uint32_t a_passesNum);
  void Collatz(uint32_t tid, uint32_t a_maxIters, const uint32_t* a_start [[size("tid")]],
               uint32_t* a_steps [[size("tid")]], uint32_t* a_peak [[size("tid")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  void kernel_Init      (uint32_t tid, const uint32_t* a_start, uint32_t* a_value, uint32_t* a_count, uint32_t* a_currPeak);
  bool kernel_Step      (uint32_t tid, uint32_t* a_value);
  void kernel_Accumulate(uint32_t tid, const uint32_t* a_value, uint32_t* a_count, uint32_t* a_currPeak);
  void kernel_Write     (uint32_t tid, const uint32_t* a_count, const uint32_t* a_currPeak, uint32_t* a_outSteps, uint32_t* a_outPeak);
};
