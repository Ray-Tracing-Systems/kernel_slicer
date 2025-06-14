#include "blur.h"
#include <algorithm>
#include <chrono>

void GaussianBlur::kernel1D_generateKernel(int kernelRadius, float sigma)
{
    float sum = 0.0f;
    for (int i = -kernelRadius; i <= kernelRadius; i++)
    {
        float value = exp(-(i * i) / (2.0f * sigma * sigma));
        m_kernel[i + kernelRadius] = value;
        sum += value;
    }
    
    // Нормализация ядра
    for (int i = 0; i < 2 * kernelRadius + 1; i++)
    {
        m_kernel[i] /= sum;
    }
}

void GaussianBlur::kernel2D_horizontalBlur(int w, int h, const float* inData, float* tempData, int kernelRadius)
{
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
            
            for (int i = -kernelRadius; i <= kernelRadius; i++)
            {
                int xi = std::min(std::max(x + i, 0), w - 1);
                float weight = m_kernel[i + kernelRadius];
                
                sum.x += inData[4 * (y * w + xi) + 0] * weight;
                sum.y += inData[4 * (y * w + xi) + 1] * weight;
                sum.z += inData[4 * (y * w + xi) + 2] * weight;
                sum.w += inData[4 * (y * w + xi) + 3] * weight;
            }
            
            tempData[4 * (y * w + x) + 0] = sum.x;
            tempData[4 * (y * w + x) + 1] = sum.y;
            tempData[4 * (y * w + x) + 2] = sum.z;
            tempData[4 * (y * w + x) + 3] = sum.w;
        }
    }
}

void GaussianBlur::kernel2D_verticalBlur(int w, int h, const float* tempData, float* outData, int kernelRadius)
{
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            float4 sum = float4(0.0f, 0.0f, 0.0f, 0.0f);
            
            for (int i = -kernelRadius; i <= kernelRadius; i++)
            {
                int yi = std::min(std::max(y + i, 0), h - 1);
                float weight = m_kernel[i + kernelRadius];
                
                sum.x += tempData[4 * (yi * w + x) + 0] * weight;
                sum.y += tempData[4 * (yi * w + x) + 1] * weight;
                sum.z += tempData[4 * (yi * w + x) + 2] * weight;
                sum.w += tempData[4 * (yi * w + x) + 3] * weight;
            }
            
            outData[4 * (y * w + x) + 0] = sum.x;
            outData[4 * (y * w + x) + 1] = sum.y;
            outData[4 * (y * w + x) + 2] = sum.z;
            outData[4 * (y * w + x) + 3] = sum.w;
        }
    }
}

void GaussianBlur::Run(int w, int h, const float* inData, float* outData, int kernelRadius, float sigma)
{
    if (w > m_maxWidth || h > m_maxHeight || kernelRadius > m_maxKernelRadius)
    {
        throw std::runtime_error("Input dimensions exceed maximum allowed size");
    }

    auto before = std::chrono::high_resolution_clock::now();
    
    kernel1D_generateKernel(kernelRadius, sigma);
    kernel2D_horizontalBlur(w, h, inData, m_tempBuffer.data(), kernelRadius);
    kernel2D_verticalBlur(w, h, m_tempBuffer.data(), outData, kernelRadius);
    
    m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count() / 1000.f;
}