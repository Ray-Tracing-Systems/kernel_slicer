#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>
#include <filesystem>

#include "ArgParser.h"
#include "render2d.h"
#include "Image2d.h"

#include "vk_context.h"
std::shared_ptr<ProcRender2D> CreateProcRender2D_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 
#include "render2d_generated.h"

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  int w = 1024,h = 1024; 
  std::vector<uint> ldrData(w*h);
  
  ArgParser args(argc, argv);
  bool onGPU  = args.hasOption("--gpu");

  std::shared_ptr<ProcRender2D> pImpl = nullptr;

  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    std::vector<const char*> requiredExtensions;
    auto deviceFeatures = ProcRender2D_Generated::ListRequiredDeviceFeatures(requiredExtensions);
    auto ctx            = vk_utils::globalContextInit(requiredExtensions, enableValidationLayers, a_preferredDeviceId, &deviceFeatures);
    //auto ctx = vk_utils::globalContextGet(false, 0);
    pImpl = CreateProcRender2D_Generated(ctx, w*h);
  }
  else
    pImpl = std::make_shared<ProcRender2D>();

  pImpl->CommitDeviceData();
  
  const char* filename = "z_stat.csv";
  bool file_exists = std::filesystem::exists(filename);
  std::ofstream fout(filename, std::ios::app);
  if(!file_exists)
  {
    fout << "Implementations;Branching;Time(cmd);Time(kernel);" << std::endl;
  }
  
  int branchingModes[3] = {int(ProcRender2D::BRANCHING_LITE), 
                           int(ProcRender2D::BRANCHING_MEDIUM), 
                           int(ProcRender2D::BRANCHING_HEAVY)};
  
  const char* branchingName[3] = {"lite", "medium", "heavy"};
  
  for(int i=0;i<3;i++) 
  for(int implNum = 1; implNum <= ProcRender2D::TOTAL_IMPLEMANTATIONS; implNum++)
  {
    pImpl->SetImplementationCount(implNum);
    pImpl->UpdateMembersPlainData();

    pImpl->Fractal(w, h, ldrData.data(), branchingModes[i]);
    
    std::stringstream strOut;
    {
      if(onGPU)
        strOut << "zout_gpu";
      else
        strOut << "zout_cpu";
      strOut << "_" << implNum << "_" << branchingName[i] << ".bmp";
    }

    std::string fileName = strOut.str();
    LiteImage::SaveBMP(fileName.c_str(), ldrData.data(), w, h);
  
    float timings[4] = {0,0,0,0};
    pImpl->GetExecutionTime("Fractal", timings);
    std::cout << "Fractal(exec) = " << timings[0]              << " ms " << std::endl;
    std::cout << "Fractal(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
    std::cout << "Fractal(ovrh) = " << timings[3]              << " ms " << std::endl;

    float timeCmd = timings[0];

    pImpl->GetExecutionTime("kernel2D_EvaluateTextures", timings);
    std::cout << "Fractal(kernel time) = " << timings[0] << " ms " << std::endl;

    float timeKern = timings[0];
    fout << implNum <<";" << branchingName[i] << ";" << timeCmd << ";" << timeKern << ";" << std::endl;
  }

  return 0;
}
