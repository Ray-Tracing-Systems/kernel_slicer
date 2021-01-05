
# Example #5: Bloom in IPV pattern

Now we consider simple example of image processing. Please note that this pattern is quite general and can be used for many different aplications other than Image Processing. But for now, let's assume that you want to implement a simple tone mapping image filter which implements "Bloom" effect. This filter blur bright regions of image and blends them with original image to simulate glow effect.

image.here

Our filter consists of several passes (fig. 1):

1. Extracting bright pixels to separate image;

2. Downsaimpling of bright pixels image. This pass is used for optimisation purposes because it allow to reduce blur size;

3. Blur downsampled image (We used 2 different passes BlurX and BlurY because gauss blur is separable filter);

4. Blend downsamples and blurred pixels with the oroginal pixels.

In fact blur can be cobmined with any advanced tone mapping filter, but in this example we used simple clamping for actual tone mapping. So, our CPU implementation consists of several functions and data members.


```cpp
class ToneMapping 
{
public:

  void Bloom(int w, int h, const float4* inData4f, unsigned int* outData1ui);

protected:

  void kernel2D_ExtractBrightPixels(int tidX, int tidY, const float4* inData4f, float4* a_brightPixels);
  void kernel2D_DownSample4x(int x, int y, const float4* a_daraFullRes, float4* a_dataSmallRes);
  void kernel2D_BlurX(int tidX, int tidY, const float4* a_dataIn, float4* a_dataOut);
  void kernel2D_BlurY(int tidX, int tidY, const float4* a_dataIn, float4* a_dataOut);
  void kernel2D_MixAndToneMap(int tidX, int tidY, const float4* inData4f, const float4* inBrightPixels, unsigned int* outData1ui);

  std::vector<float4> m_brightPixels;
  std::vector<float4> m_downsampledImage;
  std::vector<float4> m_tempImage;
  std::vector<float>  m_filterWeights;

  int m_blurRadius;
  int m_width;
  int m_height;
  int m_widthSmall;
  int m_heightSmall;
  float m_gammaInv;
};
```

Please note that our input is HDR color (float4), and the ouptut is LDR (unsigned int). Well, this is because we implement tone mapping filter. The CPU implementation of Bloom member function is straitforward:

```cpp
void ToneMapping::Bloom(int w, int h, const float4* inData4f, 
                        unsigned int* outData1ui)
{
  // (1) ExtractBrightPixels (inData4f => m_brightPixels (w,h))
  //
  kernel2D_ExtractBrightPixels(w, h, inData4f, 
                               m_brightPixels.data());

  // (2) Downsample (m_brightPixels => m_downsampledImage (w/4, h/4) )
  //
  kernel2D_DownSample4x(m_widthSmall, m_heightSmall, m_brightPixels.data(), 
                        m_downsampledImage.data());

  // (3) GaussBlur (m_downsampledImage => m_downsampledImage)
  //
  kernel2D_BlurX(m_widthSmall, m_heightSmall, m_downsampledImage.data(), 
                 m_tempImage.data()); // m_downsampledImage => m_tempImage

  kernel2D_BlurY(m_widthSmall, m_heightSmall, m_tempImage.data(), 
                 m_downsampledImage.data()); // m_tempImage => m_downsampledImage

  // (4) MixAndToneMap(inData4f, m_downsampledImage) => outData1ui
  //
  kernel2D_MixAndToneMap(w,h, inData4f, m_downsampledImage.data(), 
                         outData1ui);
}
```

You should now pay attention to a number of points, some of which are important for kernel_slicer in general and others are important particular for IPV pattern:

* (general) There are two main types of member functions --- Kernel Functions (KF) and Control Functions (CF);

* (general) Control Functions should call Kernel Functions. This is the way how Control Functions ase distinguished;

* (general) Other functions (both member and non member) can be called both from KF and CF (calling member functions inside KF is **not yet implemented!**);

* (IPV specific) Kernel Functions have one of further prefixes: "kernel1D_", "kernel2D_" or "kernel3D_". This is the way how they are distinguished. Such functions will be transformed to GPU kernels;

* (IPV specific) Kernel Functions contain thread loops inside. A prefix in the kernel name will define how many nested loops should be eliminated from kernels source code to transform it to GPU kernel;

* (IPV specific) For current implementation it is important to pass loop size via arguments of KF (we plan to remove this restriction).

Now let us see what happends with KF source code. Here is the input example:

```cpp
void ToneMapping::kernel2D_ExtractBrightPixels(int width, int height, const float4* inData4f, float4* a_brightPixels)
{  
  for(int y=0;y<height;y++)
  {
    for(int x=0;x<width;x++)
    {
      float4 pixel = inData4f[pitch(x, y, m_width)];
      if(pixel.x >= 1.0f || pixel.y >= 1.0f || pixel.z >= 1.0f)
        a_brightPixels[pitch(x, y, m_width)] = pixel;
      else
        a_brightPixels[pitch(x, y, m_width)] = make_float4(0,0,0,0);      
    }
  }
}
```

And here is the output:

```cpp
__kernel void kernel2D_ExtractBrightPixels(
  __global const float4 * restrict inData4f,
  __global float4 * restrict a_brightPixels,
  __global uint* restrict kgen_data,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  /////////////////////////////////////////////////////////////////
  const uint x = get_global_id(0); 
  const uint y = get_global_id(1); 
  if(x >= kgen_iNumElementsX || y >= kgen_iNumElementsY)
    return;
  const int m_width = *( (__global const int*)(kgen_data+12));
  /////////////////////////////////////////////////////////////////
  float4 pixel = inData4f[pitch(x, y, m_width)];
  if(pixel.x >= 1.0f || pixel.y >= 1.0f || pixel.z >= 1.0f)
    a_brightPixels[pitch(x, y, m_width)] = pixel;
  else
    a_brightPixels[pitch(x, y, m_width)] = make_float4(0,0,0,0);      
}
```
So, the transformation is general is straitforward except that we read 'm_width' from some buffer named 'kgen_data'. Well 'kgen_data' is uniform buffer in which host code should put data.
Here you can see that in generated host code offset in bytes for 'm_width' is 48, and 48/sizeof(uint) = 12 which therefore is correct index for our data member.

```cpp
void ToneMapping_Generated::UpdatePlainMembers(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
{
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 0, &m_blurRadius, 4);
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 28, &m_gammaInv, 4);
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 32, &m_height, 4);
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 36, &m_heightSmall, 4);
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 48, &m_width, 4);
  a_pCopyEngine->UpdateBuffer(m_classDataBuffer, 52, &m_widthSmall, 4);
}
```

So, now you should now that kernel_slicer will generate not just kernels, but also the whole strapping code in Vulkan which allow to run kernels. Now you should understand how to use generated code.
There are several important functions that will be generated. 

```cpp
class ToneMapping_Generated : public ToneMapping
{
public:
  virtual void InitVulkanObjects(...) { ... }
  virtual void InitMemberBuffers();
  virtual void UpdateAll(std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine) {...}

  virtual void SetVulkanInOutFor_Bloom(
    VkBuffer a_inData4fBuffer,
    size_t   a_inData4fOffset,
    VkBuffer a_outData1uiBuffer,
    size_t   a_outData1uiOffset,
    uint32_t dummyArgument = 0)
  {
    ...
  }

  virtual void BloomCmd(VkCommandBuffer a_commandBuffer, int w, int h, const float4* inData4f, 
                        unsigned int* outData1ui);
  
}
```

1. InitVulkanObjects. You should init Vulkan stuff before using other functions. This function is also set kernels configuration. In this example we used 32x8 blocks instead of 16x16 to prevent address leap inside warp on Nvidia HW:
```cpp
  auto pGPUImpl = std::make_shared<ToneMapping_Generated>();         
  pGPUImpl->InitVulkanObjects(device, physicalDevice, w*h, 32, 8, 1); 
```
2. InitMemberBuffers. This function will init all internal buffers with the current **capacity** of used std::vectors. So, you have to call it **after** all vector memebers aere initialized with the correct size by SetMaxImageSize:
```cpp
  pGPUImpl->SetMaxImageSize(w, h);                                    
  pGPUImpl->InitMemberBuffers();      
```
3. UpdateAll. This is interesting one. This function will update (CPU ==> GPU) all data members which are used in kernels via provided by used implementation of *vkfw::ICopyEngine::UpdateBuffer*. Of course, we do not know what kind of data you are going to update and how often. Therefore, all we can do is provide you with functions and an interface for updating both individual data members and all data combined. The implementation is the user responsibility but we provide simpe implementation in our examples of course. If you are goint to implement some advanced logic you have two options which can be used both separately or siumultaniously:

    1. Create another class 'ToneMapping_GPU' which inherits 'ToneMapping_Generated'. In this class you can implement any advanced uppdate logic. For example if you know that some vectors pushed new data in their end, you need to update only end of the vector (Note that GPU memory for vectors is allocated using vector.capacity() number of bytes). 

    2. Create advanced implementation of *vkfw::ICopyEngine*. For example you can collect small updates in separate host buffers and right after UpdateAll() is finished you may actually perform all such updates. 

    ```cpp
    pGPUImpl->UpdateAll(pCopyHelper);                                    
    pCopyHelper->Flush(); // assume you custom implementation, we don't have such in our examples.       
    ```  

4. SetVulkanInOutFor_Bloom. As you can see, for the control function 'Bloom' two member functions were actually generated. They are 'SetVulkanInOutFor_Bloom' and 'BloomCmd'. This is because Vulkan don't allow you to create new bindings (descriptor sets) during writing command buffer. That is, you have to create bindings in advance anyway if you use Vulkan.

```cpp
pGPUImpl->SetVulkanInOutFor_Bloom(colorBufferHDR, 0,  // ==> 
                                  colorBufferLDR, 0); // <==
``` 

5. BloomCmd. Finally you can use 'BloomCmd' to write Bloom kernel sequence to command buffer (which constist of several kernel dispatchs with apropriate dewscriptor sets bindings).

