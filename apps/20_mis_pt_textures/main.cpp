#include <iostream>
#include <fstream>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<Integrator> CreateIntegrator_Generated(int a_maxThreads, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const int WIN_WIDTH  = 1024;
  const int WIN_HEIGHT = 1024;

  std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
  std::vector<uint32_t> packedXY(WIN_WIDTH*WIN_HEIGHT);
  std::vector<float4>   realColor(WIN_WIDTH*WIN_HEIGHT);
  
  std::shared_ptr<Integrator> pImpl = nullptr;
  ArgParser args(argc, argv);

  std::string scenePath = "../resources/HydraCore/hydra_app/tests/test_42/statex_00001.xml";
  if(args.hasOption("-in"))
    scenePath = args.getOptionValue<std::string>("-in");

  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateIntegrator_Generated( WIN_WIDTH*WIN_HEIGHT, ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  else
    pImpl = std::make_shared<Integrator>(WIN_WIDTH*WIN_HEIGHT);
  
  pImpl->SetViewport(0,0,WIN_WIDTH,WIN_HEIGHT);
  
  pImpl->LoadScene(scenePath.c_str());
  pImpl->CommitDeviceData();

  // remember pitch-linear (x,y) for each thread to make our threading 1D
  //
  std::cout << "CastSingleRayBlock() ... " << std::endl; 
  pImpl->PackXYBlock(WIN_WIDTH, WIN_HEIGHT, packedXY.data(), 1);
  pImpl->CastSingleRayBlock(WIN_HEIGHT*WIN_HEIGHT, packedXY.data(), pixelData.data(), 1);
  
  if(onGPU)
    SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  // -------------------------------------------------------------------------------

  const int PASS_NUMBER = 100;
  const float normConst = 1.0f/float(PASS_NUMBER);
  const float invGamma  = 1.0f/2.2f;
  
  // now test path tracing
  //
  const int NAIVE_PT_REPEAT = 1;

  std::cout << "NaivePathTraceBlock() ... " << std::endl;
  memset(realColor.data(), 0, sizeof(float)*4*realColor.size());
  pImpl->SetIntegratorType(Integrator::INTEGRATOR_STUPID_PT);
  pImpl->UpdateMembersPlainData();
  pImpl->NaivePathTraceBlock(WIN_HEIGHT*WIN_HEIGHT, 6, packedXY.data(), realColor.data(), PASS_NUMBER*NAIVE_PT_REPEAT);
  
  float minValPdf = 1e30f;
  float maxValPdf = -1e29f;
  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    float4 color = realColor[i]*normConst*(1.0f/float(NAIVE_PT_REPEAT));
    if(std::isfinite(color.w))
    {
      minValPdf    = std::min(minValPdf, color.w);
      maxValPdf    = std::max(maxValPdf, color.w);
    }
    color.x      = std::pow(color.x, invGamma);
    color.y      = std::pow(color.y, invGamma);
    color.z      = std::pow(color.z, invGamma);
    color.w      = 1.0f;
    pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
  }
  
  if(onGPU)
    SaveBMP("zout_gpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  std::cout << "minValPdf = " << minValPdf << std::endl;
  std::cout << "maxValPdf = " << maxValPdf << std::endl;
  const float normInv = 1.0f / (maxValPdf - minValPdf);
  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    const float pdf = std::isfinite(realColor[i].w) ? (realColor[i].w*normConst*(1.0f/float(NAIVE_PT_REPEAT)) - minValPdf)*normInv : 0.0f;
    float4 color;
    color.x      = pdf;
    color.y      = std::sqrt(pdf);
    color.z      = std::sqrt(pdf);
    color.w      = 1.0f;
    pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
  }
  
  if(onGPU)
    SaveBMP("zout_gpu1.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu1.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  // -------------------------------------------------------------------------------

  std::cout << "PathTraceBlock(Shadow-PT) ... " << std::endl;
  memset(realColor.data(), 0, sizeof(float)*4*realColor.size());
  pImpl->SetIntegratorType(Integrator::INTEGRATOR_SHADOW_PT);
  pImpl->UpdateMembersPlainData();
  pImpl->PathTraceBlock(WIN_HEIGHT*WIN_HEIGHT, 6, packedXY.data(), realColor.data(), PASS_NUMBER);

  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    float4 color = realColor[i]*normConst;
    color.x      = std::pow(color.x, invGamma);
    color.y      = std::pow(color.y, invGamma);
    color.z      = std::pow(color.z, invGamma);
    color.w      = 1.0f;
    pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
  }

  if(onGPU)
    SaveBMP("zout_gpu3.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu3.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  // -------------------------------------------------------------------------------

  std::cout << "PathTraceBlock(MIS-PT) ... " << std::endl;
  memset(realColor.data(), 0, sizeof(float)*4*realColor.size());
  pImpl->SetIntegratorType(Integrator::INTEGRATOR_MIS_PT);
  pImpl->UpdateMembersPlainData();
  pImpl->PathTraceBlock(WIN_HEIGHT*WIN_HEIGHT, 6, packedXY.data(), realColor.data(), PASS_NUMBER);

  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    float4 color = realColor[i]*normConst;
    color.x      = std::pow(color.x, invGamma);
    color.y      = std::pow(color.y, invGamma);
    color.z      = std::pow(color.z, invGamma);
    color.w      = 1.0f;
    pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
  }

  if(onGPU)
    SaveBMP("zout_gpu4.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu4.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  // -------------------------------------------------------------------------------

  std::cout << std::endl;
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("NaivePathTraceBlock", timings);
  std::cout << "NaivePathTraceBlock(exec)  = " << timings[0]              << " ms " << std::endl;
  std::cout << "NaivePathTraceBlock(copy)  = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "NaivePathTraceBlock(ovrh)  = " << timings[3]              << " ms " << std::endl;
  std::cout << std::endl;
  pImpl->GetExecutionTime("PathTraceBlock", timings);
  std::cout << "PathTraceBlock(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "PathTraceBlock(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "PathTraceBlock(ovrh) = " << timings[3]              << " ms " << std::endl;
  return 0;
}

