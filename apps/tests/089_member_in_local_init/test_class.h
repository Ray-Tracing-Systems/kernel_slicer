#pragma once
#include <vector>
#include <cstdint>

// Class members inside parentheses in the initializer of a local variable of a 2D kernel,
// e.g. 'const uint32_t idx = (y + m_startY) * m_pitch + (x + m_startX);', must be accessed through the class data in shaders
class LocalInit
{
public:
  LocalInit(){}

  void SetWindow(uint32_t a_startX, uint32_t a_startY, uint32_t a_pitch) { m_startX = a_startX; m_startY = a_startY; m_pitch = a_pitch; }

  virtual void Run(uint32_t a_width, uint32_t a_height, const float* a_in [[size("a_size")]], float* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void UpdateMembersPlainData() {}                                 // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  void kernel2D_Window(uint32_t a_width, uint32_t a_height, const float* a_in, float* a_out);

  uint32_t m_startX = 0;
  uint32_t m_startY = 0;
  uint32_t m_pitch  = 0;
};
