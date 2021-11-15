#include <android/log.h>
#include <android_native_app_glue.h>
#include <unordered_map>

#ifdef NDEBUG
const bool g_enableValidationLayers = false;
#else
const bool g_enableValidationLayers = true;
#endif

#include <array>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <memory>
#include <random>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cassert>

#include "LiteMath.h"
#include "ImageLoader.h"
#include "06_n_body/test_class.h"

#define ARRAY_SUM_INT_SAMPLE 0
#define ARRAY_SUM_FLOAT_SAMPLE 1
#define NBODY_SAMPLE 2
#define SPHARM_SAMPLE 3
#define BLOOM_SAMPLE 4
#define NLM_SAMPLE 5
#define PT_SAMPLE 6
#define SAMPLE_NUMBER SPHARM_SAMPLE


static const char* tag_app = "com.slicer.compute";
#define LOGI(...) \
  ((void)__android_log_print(ANDROID_LOG_INFO, tag_app, __VA_ARGS__))
#define LOGW(...) \
  ((void)__android_log_print(ANDROID_LOG_WARN, tag_app, __VA_ARGS__))
#define LOGE(...) \
  ((void)__android_log_print(ANDROID_LOG_ERROR, tag_app, __VA_ARGS__))

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

namespace vk_android
{
  extern AAssetManager *g_pMgr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class androidbuf : public std::streambuf {
public:
  enum { bufsize = 1024 }; // ... or some other suitable buffer size
  androidbuf() { this->setp(buffer, buffer + bufsize - 1); }

private:
  int overflow(int c)
  {
    if (c == traits_type::eof()) {
      *this->pptr() = traits_type::to_char_type(c);
      this->sbumpc();
    }
    return this->sync()? traits_type::eof(): traits_type::not_eof(c);
  }

  int sync()
  {
    int rc = 0;
    if (this->pbase() != this->pptr()) {
      char writebuf[bufsize+1];
      memcpy(writebuf, this->pbase(), this->pptr() - this->pbase());
      writebuf[this->pptr() - this->pbase()] = '\0';

      rc = __android_log_write(ANDROID_LOG_INFO, tag_app, writebuf) > 0;
      this->setp(buffer, buffer + bufsize - 1);
    }
    return rc;
  }

  char buffer[bufsize];
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int32_t array_summ_cpu(const std::vector<int32_t>& array);
int32_t array_summ_gpu(const std::vector<int32_t>& array);
float array_summ_float_cpu(const std::vector<float>& array);
float array_summ_float_gpu(const std::vector<float>& array);
std::vector<nBody::BodyState> n_body_cpu(uint32_t seed, uint32_t iterations);
std::vector<nBody::BodyState> n_body_gpu(uint32_t seed, uint32_t iterations);
std::vector<LiteMath::float3> process_image_cpu(std::vector<uint32_t>& a_inPixels, uint32_t a_width, uint32_t a_height);
std::vector<LiteMath::float3> process_image_gpu(std::vector<uint32_t>& a_inPixels, uint32_t a_width, uint32_t a_height);
std::vector<uint> tone_mapping_cpu(int w, int h, float* a_hdrData);
std::vector<uint> tone_mapping_gpu(int w, int h, float* a_hdrData);
std::vector<uint>  Denoise_cpu(const int w, const int h, const float* a_hdrData, int32_t* a_inTexColor, const int32_t* a_inNormal, const float* a_inDepth,
                               const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel);
std::vector<uint>  Denoise_gpu(const int w, const int h, const float* a_hdrData, int32_t* a_inTexColor, const int32_t* a_inNormal, const float* a_inDepth,
                               const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel);

std::vector<uint32_t> test_class_gpu(); //pt

const float EPS = 1e-3f;

inline float errFloat4(const LiteMath::float4& a, const LiteMath::float4& b)
{
  return std::max(std::max(std::abs(a.x - b.x), std::abs(a.y - b.y)),
                  std::max(std::abs(a.z - b.z), std::abs(a.w - b.w)));
}

class TestApp
{
public:

  void run(android_app* a_androidAppCtx)
  {
    androidApp = a_androidAppCtx;
    app_dir = androidApp->activity->externalDataPath;
    std::cout << "DIRECTORY WITH SAVED FILES: " << app_dir << std::endl;
    switch(SAMPLE_NUMBER)
    {
      case ARRAY_SUM_INT_SAMPLE:
        std::cout << "RUNNING ARRAY_SUM_INT_SAMPLE...\n";
        ArraySumInt();
        break;
      case ARRAY_SUM_FLOAT_SAMPLE:
        std::cout << "RUNNING ARRAY_SUM_FLOAT_SAMPLE...\n";
        ArraySumFloat();
        break;
      case NBODY_SAMPLE:
        std::cout << "RUNNING NBODY_SAMPLE...\n";
        NBody();
        break;
      case SPHARM_SAMPLE:
        std::cout << "RUNNING SPHARM_SAMPLE...\n";
        Spharm();
        break;
      case BLOOM_SAMPLE:
        std::cout << "RUNNING BLOOM_SAMPLE...\n";
        Bloom();
        break;
      case NLM_SAMPLE:
        std::cout << "RUNNING NLM_SAMPLE...\n";
        Denoise();
        break;
      case PT_SAMPLE:
        std::cout << "RUNNING PT_SAMPLE...\n";
        PathTrace();
        break;
      default:
        break;
    }
  }

private:

  android_app* androidApp;
  std::string app_dir;

  void ArraySumInt()
  {
    std::vector<int32_t> array(1024 * 1024);
    for(size_t i=0;i<array.size();i++)
    {
      if(i%3 == 0)
        array[i] = i;
      else
        array[i] = -i;
    }

    auto cpu_summ = array_summ_cpu(array);
    auto gpu_summ = array_summ_gpu(array);
    std::cout << "[cpu]: array summ  = " << cpu_summ << std::endl;
    std::cout << "[gpu]: array summ  = " << gpu_summ << std::endl;
  }

  void ArraySumFloat()
  {
    std::vector<float> array(1024 * 1024);
    for(size_t i=0;i<array.size();i++)
    {
      if(i % 3 == 0)
        array[i] = i * 3.14f;
      else
        array[i] = -i * 2.72f;
    }

    auto cpu_summ = array_summ_float_cpu(array);
    auto gpu_summ = array_summ_float_gpu(array);
    std::cout << "[cpu]: array summ  = " << cpu_summ << std::endl;
    std::cout << "[gpu]: array summ  = " << gpu_summ << std::endl;
  }

  void NBody()
  {
    const uint32_t SEED       = 42;
    const uint32_t ITERATIONS = 10;
    auto out_cpu = n_body_cpu(SEED, ITERATIONS);
    auto out_gpu = n_body_gpu(SEED, ITERATIONS);
    bool failed = false;

    std::vector<uint32_t> badId; badId.reserve(10);
    std::vector<uint32_t> badId2; badId2.reserve(10);

    float maxErr   = 0.0f;
    float avgError = 0.0f;

    for (uint32_t i = 0; i < out_cpu.size(); ++i)
    {
      float errPos = errFloat4(out_cpu[i].pos_weight, out_gpu[i].pos_weight);
      float errVel = errFloat4(out_cpu[i].vel_charge, out_gpu[i].vel_charge);

      if (errPos > EPS)
      {
        if(badId.size() + 1 < badId.capacity())
          badId.push_back(i);
      }

      if (errVel > EPS)
      {
        if(badId2.size() + 1 < badId2.capacity())
          badId2.push_back(i);
      }

      avgError += std::max(errVel,errPos);
      maxErr = std::max(maxErr, std::max(errVel,errPos));
    }

    avgError /= float(out_cpu.size());

    std::cout << "maxErr = " << maxErr << std::endl;
    std::cout << "avgErr = " << avgError << std::endl;

    for(const auto i : badId)
    {
      std::cout << "Wrong position " << i << std::endl;
      std::cout << "CPU value: " << out_cpu[i].pos_weight.x << "\t" << out_cpu[i].pos_weight.y << "\t" << out_cpu[i].pos_weight.z << "\t" << out_cpu[i].pos_weight.w << std::endl;
      std::cout << "GPU value: " << out_gpu[i].pos_weight.x << "\t" << out_gpu[i].pos_weight.y << "\t" << out_gpu[i].pos_weight.z << "\t" << out_gpu[i].pos_weight.w << std::endl;
    }

    for(const auto i : badId2)
    {
      std::cout << "Wrong velocity " << i << std::endl;
      std::cout << "CPU value: " << out_cpu[i].vel_charge.x << "\t" << out_cpu[i].vel_charge.y << "\t" << out_cpu[i].vel_charge.z << "\t" << out_cpu[i].vel_charge.w << std::endl;
      std::cout << "GPU value: " << out_gpu[i].vel_charge.x << "\t" << out_gpu[i].vel_charge.y << "\t" << out_gpu[i].vel_charge.z << "\t" << out_gpu[i].vel_charge.w << std::endl;
    }

    if (failed) {
      std::cout << "FAIL" << std::endl;
    } else {
      std::cout << "OK" << std::endl;
    }
  }

  void Spharm()
  {
    std::string filename = "skybox.bmp";
    int w, h;
    std::vector<uint32_t> inputImageData = LoadBMPAndroid(filename.c_str(), &w, &h);

    std::cout << "compute ... " << std::endl;
    auto result  = process_image_cpu(inputImageData, w, h);
    auto result2 = process_image_gpu(inputImageData, w, h);

    std::cout << "save to file in " << androidApp->activity->externalDataPath << std::endl;
    {
      std::ofstream out(app_dir + "/spharm_output_cpu.bin", std::ios::binary);
      if(!out.is_open())
        std::cout << "spharm_output_cpu not open!" << std::endl;
      for (size_t i = 0; i < result.size(); ++i) {
//        out << result[i].x << result[i].y << result[i].z;
        out.write(reinterpret_cast<char*>(&result[i].x), sizeof(float));
        out.write(reinterpret_cast<char*>(&result[i].y), sizeof(float));
        out.write(reinterpret_cast<char*>(&result[i].z), sizeof(float));
      }
      out.close();

      std::ofstream out2(app_dir + "/spharm_output_gpu.bin", std::ios::binary);
      if(!out.is_open())
        std::cout << "spharm_output_gpu not open!" << std::endl;
      for (size_t i = 0; i < result2.size(); ++i) {
//        out << result2[i].x << result2[i].y << result2[i].z;
        out2.write(reinterpret_cast<char*>(&result2[i].x), sizeof(float));
        out2.write(reinterpret_cast<char*>(&result2[i].y), sizeof(float));
        out2.write(reinterpret_cast<char*>(&result2[i].z), sizeof(float));
      }
      out2.close();
    }
//
    std::cout << "output: " << std::endl;
    for(size_t i=0;i<result2.size();i++)
    {
      std::cout << "cpu: " << result[i].x  << " " << result[i].y  << " " << result[i].z  << "\n"
                << "gpu: " << result2[i].x << " " << result2[i].y << " " << result2[i].z << " " << std::endl << std::endl;
    }
  }

  void Bloom()
  {
    std::vector<float> hdrData;
    int w,h;
    if(!LoadEXRImageFromFile("nancy_church_2.exr", &w, &h, hdrData))
    {
      std::cout << "can't open input file 'nancy_church_2.exr' " << std::endl;
      return;
    }

    auto addressToCheck = reinterpret_cast<uint64_t>(hdrData.data());
    assert(addressToCheck % 16 == 0); // check if address is aligned!!!

    std::string cpu_out_name = app_dir + "/zout_cpu.bmp";
    auto cpu_out = tone_mapping_cpu(w, h, hdrData.data());
    SaveBMPAndroid(cpu_out_name.c_str(), cpu_out.data(), w, h);

    std::string gpu_out_name = app_dir + "/zout_gpu.bmp";
    auto gpu_out = tone_mapping_gpu(w, h, hdrData.data());
    SaveBMPAndroid(gpu_out_name.c_str(), gpu_out.data(), w, h);
  }

  void Denoise()
  {
    std::vector<float>   hdrData;
    std::vector<int32_t> texColor;
    std::vector<int32_t> normal;
    std::vector<float>   depth;

    int w, h, w2, h2, w3, h3, w4, h4;

    bool hasError = false;

    if(!LoadEXRImageFromFile("WasteWhite_1024sample.exr", &w, &h, hdrData))
    {
      std::cout << "can't open input file 'WasteWhite_1024sample.exr' " << std::endl;
      hasError = true;
    }

    if(!LoadEXRImageFromFile("WasteWhite_depth.exr", &w2, &h2, depth))
    {
      std::cout << "can't open input file 'WasteWhite_depth.exr' " << std::endl;
      hasError = true;
    }

    if(!LoadLDRImageFromFile("WasteWhite_diffcolor.png", &w3, &h3, texColor))
    {
      std::cout << "can't open input file 'WasteWhite_diffcolor.png' " << std::endl;
      hasError = true;
    }

    if(!LoadLDRImageFromFile("WasteWhite_normals.png", &w4, &h4, normal))
    {
      std::cout << "can't open input file 'WasteWhite_normals.png' " << std::endl;
      hasError = true;
    }


    if(w != w2 || h != h2)
    {
      std::cout << "size source image and depth pass not equal.' " << std::endl;
      hasError = true;
    }

    if(w != w3 || h != h3)
    {
      std::cout << "size source image and color pass not equal.' " << std::endl;
      hasError = true;
    }

    if(w != w4 || h != h4)
    {
      std::cout << "size source image and normal pass not equal.' " << std::endl;
      hasError = true;
    }

    if (hasError)
      return;


    uint64_t addressToCheck = reinterpret_cast<uint64_t>(hdrData.data());
    assert(addressToCheck % 16 == 0); // check if address is aligned!!!

    addressToCheck = reinterpret_cast<uint64_t>(depth.data());
    assert(addressToCheck % 16 == 0); // check if address is aligned!!!
//
    const int   windowRadius = 7;
    const int   blockRadius  = 3;
    const float noiseLevel   = 0.1F;

//    auto res_cpu = Denoise_cpu(w, h, hdrData.data(), texColor.data(), normal.data(), depth.data(), windowRadius, blockRadius, noiseLevel);
    auto res_gpu = Denoise_gpu(w, h, hdrData.data(), texColor.data(), normal.data(), depth.data(), windowRadius, blockRadius, noiseLevel);
//    std::string cpu_out_name = app_dir + "/denoise_cpu.bmp";
//    SaveBMPAndroid(cpu_out_name.c_str(), res_cpu.data(), w, h);

    std::string gpu_out_name = app_dir + "/denoise_gpu.bmp";
    SaveBMPAndroid(gpu_out_name.c_str(), res_gpu.data(), w, h);
  }

  void PathTrace()
  {
    auto gpu_res = test_class_gpu();

    return;
  }
};


void handle_cmd(android_app* app, int32_t cmd)
{
  assert(app->userData != NULL);
  TestApp* vk_base_app = reinterpret_cast<TestApp*>(app->userData);
  switch (cmd)
  {
    case APP_CMD_INIT_WINDOW:
      // The window is being shown, get it ready.
      vk_base_app->run(app);
      break;
    case APP_CMD_TERM_WINDOW:
      // The window is being hidden or closed, clean it up.
      //vk_app->Cleanup();
//    vk_app = nullptr;
      break;
    default:
      LOGI("event not handled: %d", cmd);
  }
}


void android_main(struct android_app* app)
{
  TestApp vk_app;
  app->userData = &vk_app;
  std::cout.rdbuf(new androidbuf);

  {
    AConfiguration *config = AConfiguration_new();
    AConfiguration_fromAssetManager(config, app->activity->assetManager);
    vk_android::g_pMgr = app->activity->assetManager;
    AConfiguration_delete(config);
  }
  {
    app->onAppCmd = handle_cmd;
    int events;
    android_poll_source *source;
    do
    {
      if (ALooper_pollAll(1, nullptr, &events, (void **) &source) >= 0)
      {
        if(source)
        {
          source->process(app, source);
        }
      }
    } while (app->destroyRequested == 0);
  }

  delete std::cout.rdbuf(nullptr);
}