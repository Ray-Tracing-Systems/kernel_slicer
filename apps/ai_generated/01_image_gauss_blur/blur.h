#pragma once

#include "LiteMath.h"
#include <vector>
#include <cmath>

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;

class GaussianBlur
{
public:
    GaussianBlur(int maxWidth, int maxHeight) : 
        m_maxWidth(maxWidth), 
        m_maxHeight(maxHeight)
    {
        // Резервируем память для временных буферов
        m_tempBuffer.resize(m_maxWidth * m_maxHeight * 4);
        m_kernel.resize(2 * m_maxKernelRadius + 1);
    }
    
    virtual void Run(int w, int h, const float* inData [[size("w*h*4")]], float* outData [[size("w*h*4")]], int kernelRadius, float sigma);
    
    virtual void CommitDeviceData() {}
    virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; }

protected:
    virtual void kernel1D_generateKernel(int kernelRadius, float sigma);
    virtual void kernel2D_horizontalBlur(int w, int h, const float* inData, float* tempData, int kernelRadius);
    virtual void kernel2D_verticalBlur(int w, int h, const float* tempData, float* outData, int kernelRadius);

    std::vector<float> m_kernel;
    std::vector<float> m_tempBuffer;
    int m_maxWidth;
    int m_maxHeight;
    const int m_maxKernelRadius = 32; // Максимальный радиус ядра
    float m_time;
};