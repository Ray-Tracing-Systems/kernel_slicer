#include "wk_context.h"
#include <iostream>
#include <cassert>
#include <cstring>

void wk_utils::printDeviceInfo(WGPUAdapter adapter) 
{
  #if WGPU_DISTR >= 30
  #else
  // Получаем свойства адаптера
  WGPUAdapterProperties props = {};
  wgpuAdapterGetProperties(adapter, &props);
  // Печатаем имя адаптера (драйвер/видеокарта)
  std::cout << "[wgpu]: Name      : " << props.name << std::endl;
  if(props.driverDescription != nullptr)
  std::cout << "[wgpu]: Driver    : " << props.driverDescription << std::endl;
  std::cout << "[wgpu]: Backend   : " << props.backendType << std::endl;
  std::cout << "[wgpu]: Device ID : " << props.deviceID << std::endl;
  #endif
}

struct UserData 
{
  WGPUAdapter adapter = nullptr;
  bool requestEnded = false;
};

#if WGPU_DISTR >= 30

static void onDeviceError(WGPUDevice const * device, WGPUErrorType type, WGPUStringView message, void* userdata1, void* userdata2) 
{
  std::string tempMsg(message.data, message.length);
  std::cout << "[wgpu device error]: " << tempMsg.c_str() << std::endl;
}

static void onAdapterRequestEnded(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* pUserData,  void *userdata2) 
{
  UserData* userData = reinterpret_cast<UserData*>(pUserData);
  if (status == WGPURequestAdapterStatus_Success) {
    userData->adapter = adapter;
  } else {
    std::string tempMsg(message.data, message.length);
    std::cout << "Could not get WebGPU adapter: " << tempMsg.c_str() << std::endl; //  << message << std::endl;
  }
  userData->requestEnded = true;
}
#else

static void onDeviceError(WGPUErrorType type, const char* message, void*) 
{
  std::cout << "[wgpu device error]: " << message << std::endl;
}

static void onAdapterRequestEnded(WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* pUserData) 
{
  UserData* userData = reinterpret_cast<UserData*>(pUserData);
  if (status == WGPURequestAdapterStatus_Success) {
      userData->adapter = adapter;
  } else {
      std::cout << "Could not get WebGPU adapter: " << message << std::endl;
  }
  userData->requestEnded = true;
}
#endif

static WGPUAdapter requestAdapterSync(WGPUInstance instance, WGPURequestAdapterOptions const* options) 
{
  UserData userData;
  #if WGPU_DISTR >= 30
  //static auto cCallback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const * message, void * userdata) -> void {
	//	onAdapterRequestEnded(status, adapter, message, userdata);
	//};
  WGPURequestAdapterCallbackInfo callbackInfo = {};
  callbackInfo.callback  = onAdapterRequestEnded;
  callbackInfo.userdata1 = &userData;
  wgpuInstanceRequestAdapter(instance, options, callbackInfo);
  #else
  wgpuInstanceRequestAdapter(instance, options, onAdapterRequestEnded, &userData);
  #endif

  // Wait until the callback sets requestEnded to true
  while (!userData.requestEnded) {
      // You may want to yield or sleep a bit here in real code
  }
  assert(userData.adapter != nullptr && "Failed to acquire adapter");
  return userData.adapter;
}

static WGPUDevice requestDeviceSync(WGPUAdapter adapter, const WGPUDeviceDescriptor* descriptor) 
{
  struct UserData {
      WGPUDevice device = nullptr;
      bool requestEnded = false;
  } userData;
  
  #if WGPU_DISTR >= 30
  auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* pUserData, void *userdata2) {
    UserData* userData = reinterpret_cast<UserData*>(pUserData);
    if (status == WGPURequestDeviceStatus_Success) {
        userData->device = device;
    } else {
     std::string tempMsg(message.data, message.length);
     std::cout << "Could not get WebGPU device: " << tempMsg.c_str() << std::endl;
    }
    userData->requestEnded = true;
  };
  WGPURequestDeviceCallbackInfo callbackInfo = {};
  callbackInfo.callback  = onDeviceRequestEnded;
  callbackInfo.userdata1 = &userData;
  wgpuAdapterRequestDevice(adapter, descriptor, callbackInfo);
  #else
  auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice device, const char* message, void* pUserData) {
    UserData* userData = reinterpret_cast<UserData*>(pUserData);
    if (status == WGPURequestDeviceStatus_Success)
      userData->device = device;
    else 
      std::cout << "Could not get WebGPU device: " << message << std::endl;
    userData->requestEnded = true;
  };
  wgpuAdapterRequestDevice(adapter, descriptor, onDeviceRequestEnded, &userData);
  #endif
  while (!userData.requestEnded) {
      // Optionally sleep/yield here
  }
  assert(userData.device != nullptr && "Failed to acquire device");
  return userData.device;
}

wk_utils::WulkanContext wk_utils::globalContextInit(WulkanDeviceFeatures a_features)
{
  WulkanContext res = {};

  // The vector size
  WGPUInstanceDescriptor desc = {};
  desc.nextInChain = nullptr;

  // We create the instance using this descriptor
  res.instance = wgpuCreateInstance(&desc);

  if (!res.instance) 
  {
    std::cout << "[wk_utils]: could not initialize WebGPU!" << std::endl;
    return res;
  }

  WGPURequestAdapterOptions adapterOpts = {};
  adapterOpts.nextInChain     = nullptr;
  adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

  res.physicalDevice = requestAdapterSync(res.instance, &adapterOpts);
  printDeviceInfo(res.physicalDevice);

  //  Create device
  WGPUDeviceDescriptor deviceDesc = {};
  deviceDesc.nextInChain = nullptr;
  #if WGPU_DISTR >= 30
  deviceDesc.label = {"Wulkan Device", WGPU_STRLEN},  // Optional: for debugging
  #else
  deviceDesc.label = "Wulkan Device"; // Optional: for debugging
  #endif
  //deviceDesc.requiredFeaturesCount = 0; // No special features
  deviceDesc.requiredFeatures = nullptr;
  deviceDesc.requiredLimits = nullptr;
  deviceDesc.defaultQueue.nextInChain = nullptr;
  #if WGPU_DISTR >= 30
  deviceDesc.defaultQueue.label = {"DefaultQueue", WGPU_STRLEN}; 
  WGPUUncapturedErrorCallbackInfo uncapturedCallbackInfo = {};
  {
    uncapturedCallbackInfo.callback  = onDeviceError;
    uncapturedCallbackInfo.userdata1 = nullptr; 
  }
  deviceDesc.uncapturedErrorCallbackInfo = uncapturedCallbackInfo;
  #else
  deviceDesc.defaultQueue.label = "DefaultQueue";
  #endif

  res.device = requestDeviceSync(res.physicalDevice, &deviceDesc);
  
  #if WGPU_DISTR <= 20
  wgpuDeviceSetUncapturedErrorCallback(res.device, onDeviceError, nullptr);
  #endif

  return res;
}

void wk_utils::readBufferBack(WulkanContext a_ctx, WGPUQueue a_queue, WGPUBuffer a_buffer, WGPUBuffer a_tmpBuffer, size_t a_size, void* a_data)
{
  WGPUCommandEncoder encoderRB = wgpuDeviceCreateCommandEncoder(a_ctx.device, nullptr);
  wgpuCommandEncoderCopyBufferToBuffer(encoderRB, a_buffer, 0, a_tmpBuffer, 0, a_size);

  WGPUCommandBuffer cmdRB = wgpuCommandEncoderFinish(encoderRB, nullptr);
  #ifdef USE_DAWN
  wgpuCommandEncoderRelease(encoderRB); //removed function ?
  #endif
  wgpuQueueSubmit(a_queue, 1, &cmdRB);
  
  // 10. Map and read back result
  struct Context {
    bool ready;
    WGPUBuffer buffer;
  };
  Context context = { false, a_tmpBuffer };
  
  #if WGPU_DISTR >= 30
  auto onBuffer2Mapped = [](WGPUMapAsyncStatus status, WGPUStringView message, void* pUserData, void* userdata2) {
    Context* context = reinterpret_cast<Context*>(pUserData);
    context->ready = true;
    if (status != WGPUMapAsyncStatus_Success) 
    {
      std::string tempMsg(message.data, message.length);
      std::cout << "[wk_utils::readBufferBack]: buffer mapped with status " << tempMsg.c_str() << std::endl;
      return;
    }
  };

  WGPUBufferMapCallbackInfo cbInfo = {};
  cbInfo.callback  = onBuffer2Mapped;
  cbInfo.userdata1 = &context;
  wgpuBufferMapAsync(a_tmpBuffer, WGPUMapMode_Read, 0, a_size, cbInfo);
   
  #else

  auto onBuffer2Mapped = [](WGPUBufferMapAsyncStatus status, void* pUserData) {
    Context* context = reinterpret_cast<Context*>(pUserData);
    context->ready = true;
    if (status != WGPUBufferMapAsyncStatus_Success) 
    {
      std::cout << "[wk_utils::readBufferBack]: buffer mapped with status " << status << std::endl;
      return;
    }
  };

  wgpuBufferMapAsync(a_tmpBuffer, WGPUMapMode_Read, 0, a_size, onBuffer2Mapped, (void*)&context);
  #endif

  while (!context.ready) 
  {
    #ifdef USE_DAWN
    wgpuDeviceTick(a_ctx.device);
    #else
    wgpuDevicePoll(a_ctx.device, false, nullptr);
    #endif
  }

  const float* mapped = static_cast<const float*>(wgpuBufferGetConstMappedRange(a_tmpBuffer, 0, a_size));
  std::memcpy(a_data, mapped, a_size);
  wgpuBufferUnmap(a_tmpBuffer);
}