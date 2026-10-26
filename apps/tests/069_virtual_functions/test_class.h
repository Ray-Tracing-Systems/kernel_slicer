#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
using LiteMath::float2;
using LiteMath::length;

// Virtual functions hierarchy (VFH) in kernel code: objects of derived types are stored in std::vector of the base type,
// a kernel calls virtual functions via pointer to an element; the type is identified by the tag
struct IShape
{
  static constexpr uint32_t TAG_EMPTY  = 0;   // !!! #REQUIRED by kernel slicer: Empty/Default impl must have zero both tag and offset
  static constexpr uint32_t TAG_CIRCLE = 1;
  static constexpr uint32_t TAG_SQUARE = 2;
  static constexpr uint32_t TAG_RING   = 3;

  IShape(){}  // dispatching on GPU hierarchy must not have destructors, especially virtual

  virtual uint32_t GetTag()          const { return TAG_EMPTY; };
  virtual float    Eval(float2 a_p)  const { return 0.0f; };

  float    m_center[2];
  float    m_size;
  uint32_t m_tag;
};

class VirtualShapes
{
public:
  VirtualShapes(){}

  void InitShapes();

  virtual void Render(uint32_t a_width, uint32_t a_height, float* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  void kernel2D_Render(uint32_t a_width, uint32_t a_height, float* a_out);

  std::vector<IShape> m_shapes;
};

struct EmptyShape : public IShape
{
  EmptyShape() { m_center[0] = 0.0f; m_center[1] = 0.0f; m_size = 0.0f; m_tag = GetTag(); }
  ~EmptyShape() = delete;

  uint32_t GetTag()         const override { return TAG_EMPTY; }
  float    Eval(float2 a_p) const override { return 0.0f; }
};

struct Circle : public IShape
{
  Circle(float a_x, float a_y, float a_radius) { m_center[0] = a_x; m_center[1] = a_y; m_size = a_radius; m_tag = GetTag(); }
  ~Circle() = delete;

  uint32_t GetTag() const override { return TAG_CIRCLE; }
  float    Eval(float2 a_p) const override
  {
    const float d = length(a_p - float2(m_center[0], m_center[1]));
    return (d < m_size) ? 1.0f - d/m_size : 0.0f;
  }
};

struct Square : public IShape
{
  Square(float a_x, float a_y, float a_halfSize) { m_center[0] = a_x; m_center[1] = a_y; m_size = a_halfSize; m_tag = GetTag(); }
  ~Square() = delete;

  uint32_t GetTag() const override { return TAG_SQUARE; }
  float    Eval(float2 a_p) const override
  {
    const float dx = std::abs(a_p.x - m_center[0]);
    const float dy = std::abs(a_p.y - m_center[1]);
    return (dx < m_size && dy < m_size) ? 0.5f : 0.0f;
  }
};

struct Ring : public IShape
{
  Ring(float a_x, float a_y, float a_radius) { m_center[0] = a_x; m_center[1] = a_y; m_size = a_radius; m_tag = GetTag(); }
  ~Ring() = delete;

  uint32_t GetTag() const override { return TAG_RING; }
  float    Eval(float2 a_p) const override
  {
    const float d = length(a_p - float2(m_center[0], m_center[1]));
    return (std::abs(d - m_size) < 0.1f) ? 0.25f + d : 0.0f;
  }
};
