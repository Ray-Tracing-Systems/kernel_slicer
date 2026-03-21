#include "test_class.h"

using namespace LiteMath;

SimpleTest::SimpleTest() {}

void SimpleTest::CalcAndAccum(uint32_t a_threadsNum, float* a_out, uint32_t a_size)
{
  kernel1D_CalcAndAccum(a_threadsNum, a_out, a_size);
}

void SimpleTest::kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out, uint32_t a_size)
{
  for(int i=0; i < a_threadsNum; i++)
  {
    float4 pointIn  = float4(1.0f + float(i), -2.0f - float(i), 3.0f + float(i)*0.5f, 1.0f );
    float4 pointOut = float4(0,0,0,0);

    float3 point3In  = float3(1.0f + float(i), -2.0f - float(i), 3.0f + float(i)*0.5f);
    float3 point3Out = float3(0,0,0);

    switch(i)
    {
      case 0:
      {
        float4x4 mt = translate4x4(float3(1,2,3));
        pointOut = mt*pointIn;
      }
      break;

      case 1:
      {
        float4x4 mt = rotate4x4X(DEG_TO_RAD*30.0f);
        pointOut = mt*pointIn;
      }
      break;

      case 2:
      {
        float4x4 mt = rotate4x4Y(DEG_TO_RAD*30.0f);
        pointOut = mt*pointIn;
      }
      break;

      case 3:
      {
        float4x4 mt = rotate4x4Z(DEG_TO_RAD*30.0f);
        pointOut = mt*pointIn;
      }
      break;

      case 4:
      {
        float3x3 mt = rotate3x3X(DEG_TO_RAD*30.0f);
        point3Out = mt*point3In;
      }
      break;

      case 5:
      {
        float3x3 mt = rotate3x3Y(DEG_TO_RAD*30.0f);
        point3Out = mt*point3In;
      }
      break;

      case 6:
      {
        float3x3 mt = rotate3x3Z(DEG_TO_RAD*30.0f);
        point3Out = mt*point3In;
      }
      break;

      default:
      break;
    };

    if(i <= 3)
    {
      a_out[4*i+0] = pointOut.x;
      a_out[4*i+1] = pointOut.y;
      a_out[4*i+2] = pointOut.z;
      a_out[4*i+3] = pointOut.w;
    }
    else
    {
      a_out[4*i+0] = point3Out.x;
      a_out[4*i+1] = point3Out.y;
      a_out[4*i+2] = point3Out.z;
      a_out[4*i+3] = 0.0f;
    }
  }
}