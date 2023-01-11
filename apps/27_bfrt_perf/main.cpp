#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<TestClass> CreateTestClass_Generated(int w, int h, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 

struct SceneSetup
{
  std::string name;
  int TC;
  int NC;
};

std::vector<std::string> split(const std::string& source, const std::string& delimiter)
{
  std::vector<std::string> result;
  size_t last = 0;
  size_t next = 0;
  while ((next = source.find(delimiter, last)) != std::string::npos)
  { 
    result.push_back(source.substr(last, next - last));
    last = next + delimiter.length();
  } 
  result.push_back(source.substr(last));
  return result;
}

std::vector<SceneSetup> LoadCSV(const char* cvsPath)
{
  std::vector<SceneSetup> res;
  res.reserve(20);
  
  std::ifstream csvread(cvsPath);
  std::string s;
  while (std::getline(csvread, s, '\n'))
  {
    auto line = split(s,";");
    if(line[0] == "Scene" || line[7] == "NC" || line[10] == "TC")
      continue;

    SceneSetup scene;
    scene.name = line[0];
    scene.NC   = std::ceil(std::stof(line[7]));
    scene.TC   = std::ceil(std::stof(line[10]));
    if(scene.NC % 2 != 0)
      scene.NC++;
    res.push_back(scene);
  }

  return res;
}


int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<TestClass> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestClass_Generated(WIN_WIDTH, WIN_HEIGHT, ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  else
    pImpl = std::make_shared<TestClass>(WIN_WIDTH, WIN_HEIGHT);
  
  std::vector<uint> pixelData(WIN_WIDTH*WIN_HEIGHT);  

  bool runBenchbark = true;
  
  if(runBenchbark)
  {
    auto scenes = LoadCSV("w_stats_eye_l.csv");
    int maxTC = 5;
    int maxNC = 10;
    for(const auto& scene : scenes) {
      maxTC = std::max(maxTC, scene.TC);
      maxNC = std::max(maxNC, scene.NC);
    }

    pImpl->InitBoxesAndTris(maxNC,maxTC); // init with some max number
    for(const auto& scene : scenes) {
      pImpl->SetBoxTrisNum(scene.NC, scene.TC);
      pImpl->CommitDeviceData();
      
      float timings[4] = {0,0,0,0};

      pImpl->BFRT_ReadAndComputeBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);
      pImpl->GetExecutionTime("BFRT_ReadAndCompute", timings);
      float timeMs_v1 = timings[0];

      pImpl->BFRT_ComputeBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);
      pImpl->GetExecutionTime("BFRT_Compute", timings);
      float timeMs_v2 = timings[0];

      float perf1 = float(WIN_WIDTH*WIN_HEIGHT)*1000.0f/(timeMs_v1*1e6f);
      float perf2 = float(WIN_WIDTH*WIN_HEIGHT)*1000.0f/(timeMs_v2*1e6f);

      std::cout << scene.name.c_str() << "; time(ms) = (" << timeMs_v1 << "," << timeMs_v2 << "); M(Rays/sec) = (" << perf1 << "," << perf2 << ")" << std::endl;
    }

  }
  else
  {
    pImpl->InitBoxesAndTris(60,25); 
    pImpl->CommitDeviceData();
  
    pImpl->BFRT_ReadAndComputeBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);
    
    if(onGPU)
      SaveBMP("zout_gpu_v1.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
    else
      SaveBMP("zout_cpu_v1.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
    float timings[4] = {0,0,0,0};
    pImpl->GetExecutionTime("BFRT_ReadAndCompute", timings);
    std::cout << "BFRT_ReadAndCompute(exec) = " << timings[0] << " ms " << std::endl;
    
    pImpl->BFRT_ComputeBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);
  
    if(onGPU)
      SaveBMP("zout_gpu_v2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
    else
      SaveBMP("zout_cpu_v2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
    pImpl->GetExecutionTime("BFRT_Compute", timings);
    std::cout << "BFRT_Compute(exec) = " << timings[0] << " ms " << std::endl;
  }

  pImpl = nullptr;
  return 0;
}