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
#include "06_n_body/test_class.h"

#define ARRAY_SUM_SAMPLE 1
#define NBODY_SAMPLE 2
#define SAMPLE_NUMBER NBODY_SAMPLE


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
std::vector<nBody::BodyState> n_body_cpu(uint32_t seed, uint32_t iterations);
std::vector<nBody::BodyState> n_body_gpu(uint32_t seed, uint32_t iterations);

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

    switch(SAMPLE_NUMBER)
    {
      case ARRAY_SUM_SAMPLE:
        std::cout << "RUNNING ARRAY_SUM_SAMPLE...\n";
        ArraySum();
        break;
      case NBODY_SAMPLE:
        std::cout << "RUNNING NBODY_SAMPLE...\n";
        NBody();
        break;
      default:
        break;
    }
  }

private:

  android_app* androidApp;

  void ArraySum()
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