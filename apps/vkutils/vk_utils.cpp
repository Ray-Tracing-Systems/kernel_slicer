#include "vk_utils.h"

#include <cstring>
#include <set>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <array>

#ifdef __ANDROID__
#include "android_native_app_glue.h"

namespace vk_android
{
  AAssetManager *g_pMgr = nullptr;
}
#endif

namespace vk_utils {

  static const char *g_debugReportExtName = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;

  std::string errorString(VkResult errorCode)
  {
    switch (errorCode)
    {
#define STR(r) case VK_##r: return #r
      STR(NOT_READY);
      STR(TIMEOUT);
      STR(EVENT_SET);
      STR(EVENT_RESET);
      STR(INCOMPLETE);
      STR(ERROR_OUT_OF_HOST_MEMORY);
      STR(ERROR_OUT_OF_DEVICE_MEMORY);
      STR(ERROR_INITIALIZATION_FAILED);
      STR(ERROR_DEVICE_LOST);
      STR(ERROR_MEMORY_MAP_FAILED);
      STR(ERROR_LAYER_NOT_PRESENT);
      STR(ERROR_EXTENSION_NOT_PRESENT);
      STR(ERROR_FEATURE_NOT_PRESENT);
      STR(ERROR_INCOMPATIBLE_DRIVER);
      STR(ERROR_TOO_MANY_OBJECTS);
      STR(ERROR_FORMAT_NOT_SUPPORTED);
      STR(ERROR_SURFACE_LOST_KHR);
      STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
      STR(SUBOPTIMAL_KHR);
      STR(ERROR_OUT_OF_DATE_KHR);
      STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
      STR(ERROR_VALIDATION_FAILED_EXT);
      STR(ERROR_INVALID_SHADER_NV);
#undef STR
      default:
        return "UNKNOWN_ERROR";
    }
  };

  void setLogToFile(const std::string &path)
  {
    FILE* log_fd = fopen( path.c_str(), "w" );
    if(!log_fd)
    {
      std::perror("[setLogToFile] File opening failed, logging to stderr");
    }
    else
    {
      log = log_fd;
    }
  }

  void runTimeError(const char* file, int line, const char* msg)
  {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_ERROR, "vk_utils", "Runtime error at %s, line %d : %s", file, line, msg);
#else
    fprintf(log, "Runtime error at %s, line %d : %s", file, line, msg);
    fflush(log);
#endif
    exit(99);
  }

  void logWarning(const std::string& msg)
  {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_WARN, "vk_utils", "Warning : %s", msg.c_str());
#else
    fprintf(log, "Warning : %s", msg.c_str());
#endif
  }

  void logInfo(const std::string& msg)
  {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "vk_utils", "%s", msg.c_str());
#else
    fprintf(log, "Info : %s", msg.c_str());
#endif
  }

  bool checkDeviceExtensionSupport(VkPhysicalDevice device, std::vector<const char *> &requestedExtensions)
  {
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(requestedExtensions.begin(), requestedExtensions.end());

    for (const auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
  }

  bool checkLayerSupport(std::vector<const char *> &requestedLayers, std::vector<std::string> &supportedLayers)
  {
    bool all_layers_supported = true;
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> layerProperties(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());
    std::vector<std::string> allPresentLayers;
    allPresentLayers.reserve(layerCount);
    for (const auto &layerProp : layerProperties)
    {
      allPresentLayers.emplace_back(layerProp.layerName);
    }

    for (const auto &layer : requestedLayers)
    {
      auto found = std::find(allPresentLayers.begin(), allPresentLayers.end(), layer);
      if (found != allPresentLayers.end())
      {
        supportedLayers.emplace_back(layer);
      }
      else
      {
        std::stringstream ss;
        ss << "Requested layer " << layer << " not found";
        logWarning(ss.str());

        all_layers_supported = false;
      }
    }

    return all_layers_supported;
  }

  VkInstance createInstance(bool &a_enableValidationLayers, std::vector<const char *> &a_requestedLayers,
    std::vector<const char *> &a_instanceExtensions, VkApplicationInfo *appInfo)
  {
    std::vector<const char *> enabledExtensions = a_instanceExtensions;
    std::vector<std::string> supportedLayers;

    checkLayerSupport(a_requestedLayers, supportedLayers);
    if (a_enableValidationLayers && !supportedLayers.empty())
    {
      uint32_t extensionCount;

      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
      std::vector<VkExtensionProperties> extensionProperties(extensionCount);
      vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensionProperties.data());

      bool foundExtension = false;
      for (VkExtensionProperties prop : extensionProperties)
      {
        if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0)
        {
          foundExtension = true;
          break;
        }
      }

      if (!foundExtension)
        RUN_TIME_ERROR("Validation layers requested but extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");

      enabledExtensions.push_back(g_debugReportExtName);
    }

    VkApplicationInfo applicationInfo = {};
    if (appInfo == nullptr)
    {
      applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
      applicationInfo.pApplicationName = "Default App Name";
      applicationInfo.applicationVersion = 0;
      applicationInfo.pEngineName = "DefaultEngine";
      applicationInfo.engineVersion = 0;
      applicationInfo.apiVersion = VK_API_VERSION_1_1;
    }
    else
    {
      applicationInfo.sType = appInfo->sType;
      applicationInfo.pApplicationName = appInfo->pApplicationName;
      applicationInfo.applicationVersion = appInfo->applicationVersion;
      applicationInfo.pEngineName = appInfo->pEngineName;
      applicationInfo.engineVersion = appInfo->engineVersion;
      applicationInfo.pNext = appInfo->pNext;
      applicationInfo.apiVersion = appInfo->apiVersion;
    }

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;

    std::vector<const char *> layer_names;
    if (a_enableValidationLayers && !supportedLayers.empty())
    {
      for (const auto &layer : supportedLayers)
        layer_names.push_back(layer.c_str());

      createInfo.enabledLayerCount = uint32_t(layer_names.size());
      createInfo.ppEnabledLayerNames = layer_names.data();

#ifndef __ANDROID__
      VkValidationFeaturesEXT validationFeatures = {};
      validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
      validationFeatures.enabledValidationFeatureCount = 1;
      VkValidationFeatureEnableEXT enabledValidationFeatures[1] = { VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT };
      validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;

//      validationFeatures.pNext = createInfo.pNext;
      createInfo.pNext = &validationFeatures;
#endif
    }
    else
    {
      createInfo.enabledLayerCount = 0;
      createInfo.ppEnabledLayerNames = nullptr;
      a_enableValidationLayers = false;
    }

    createInfo.enabledExtensionCount = uint32_t(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    VkInstance instance;
    VK_CHECK_RESULT(vkCreateInstance(&createInfo, nullptr, &instance));

    return instance;
  }


  void initDebugReportCallback(VkInstance a_instance, DebugReportCallbackFuncType a_callback, VkDebugReportCallbackEXT *a_debugReportCallback)
  {
    VkDebugReportCallbackCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
    createInfo.flags = VK_DEBUG_REPORT_INFORMATION_BIT_EXT | VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
    createInfo.pfnCallback = a_callback;

    auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(a_instance, "vkCreateDebugReportCallbackEXT");
    if (vkCreateDebugReportCallbackEXT == nullptr)
      RUN_TIME_ERROR("Could not load vkCreateDebugReportCallbackEXT");

    VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(a_instance, &createInfo, nullptr, a_debugReportCallback));
  }

  VkPhysicalDevice findPhysicalDevice(VkInstance a_instance, bool a_printInfo, unsigned a_preferredDeviceId, std::vector<const char *> a_deviceExt)
  {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(a_instance, &deviceCount, nullptr);
    if (deviceCount == 0)
    {
      RUN_TIME_ERROR("vk_utils::findPhysicalDevice, no Vulkan devices found");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(a_instance, &deviceCount, devices.data());

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    if (a_printInfo)
      std::cout << "findPhysicalDevice: { " << std::endl;

    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures features;

    for (size_t i = 0; i < devices.size(); i++)
    {
      vkGetPhysicalDeviceProperties(devices[i], &props);
      vkGetPhysicalDeviceFeatures(devices[i], &features);

      if (a_printInfo)
        std::cout << "  device " << i << ", name = " << props.deviceName;

      if (i == a_preferredDeviceId)
      {
        if (checkDeviceExtensionSupport(devices[i], a_deviceExt))
        {
          physicalDevice = devices[i];
          std::cout << " <-- (selected)" << std::endl;
        }
        else
        {
          std::cout << " <-- preferred device does not support requested extensions. Trying to find another device..." << std::endl;
        }
      }

      std::cout << std::endl;
    }

    if (a_printInfo)
      std::cout << "}" << std::endl;

    // try to select some device if preferred was not selected
    //
    if (physicalDevice == VK_NULL_HANDLE)
    {
      for (size_t i = 0; i < devices.size(); ++i)
      {
        if (checkDeviceExtensionSupport(devices[i], a_deviceExt))
        {
          physicalDevice = devices[i];
          break;
        }
      }
    }

    if (physicalDevice == VK_NULL_HANDLE)
      RUN_TIME_ERROR("vk_utils::findPhysicalDevice, no Vulkan devices supporting requested extensions were found");

    return physicalDevice;
  }

  uint32_t getQueueFamilyIndex(VkPhysicalDevice a_physicalDevice, VkQueueFlags a_bits)
  {
    uint32_t queueFamilyCount;

    vkGetPhysicalDeviceQueueFamilyProperties(a_physicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(a_physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t i = 0;
    for (; i < queueFamilies.size(); ++i)
    {
      VkQueueFamilyProperties props = queueFamilies[i];

      if (props.queueCount > 0 && (props.queueFlags & a_bits))
        break;
    }

    if (i == queueFamilies.size())
      RUN_TIME_ERROR(" vk_utils::GetComputeQueueFamilyIndex: could not find a queue family that supports operations");

    return i;
  }


  VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const std::vector<const char *> &a_enabledLayers,
                               std::vector<const char *> a_extensions, VkPhysicalDeviceFeatures a_deviceFeatures,
                               QueueFID_T &a_queueIDXs, VkQueueFlags requestedQueueTypes, void* pNextFeatures)
  {
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos{};
    const float defaultQueuePriority {0.0f};

    // Graphics queue
    if (requestedQueueTypes & VK_QUEUE_GRAPHICS_BIT)
    {
      a_queueIDXs.graphics = getQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT);
      VkDeviceQueueCreateInfo queueInfo{};
      queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueInfo.queueFamilyIndex = a_queueIDXs.graphics;
      queueInfo.queueCount = 1;
      queueInfo.pQueuePriorities = &defaultQueuePriority;
      queueCreateInfos.push_back(queueInfo);
    }
    else
    {
      a_queueIDXs.graphics = 0; //VK_NULL_HANDLE;
    }

    // Dedicated compute queue
    if (requestedQueueTypes & VK_QUEUE_COMPUTE_BIT)
    {
      a_queueIDXs.compute = getQueueFamilyIndex(physicalDevice, VK_QUEUE_COMPUTE_BIT);
      if (a_queueIDXs.compute != a_queueIDXs.graphics || queueCreateInfos.empty())
      {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = a_queueIDXs.compute;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &defaultQueuePriority;
        queueCreateInfos.push_back(queueInfo);
      }
    }
    else
    {
      a_queueIDXs.compute = a_queueIDXs.graphics;
    }

    // Dedicated transfer queue
    if (requestedQueueTypes & VK_QUEUE_TRANSFER_BIT)
    {
      a_queueIDXs.transfer = getQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT);
      if (((a_queueIDXs.transfer != a_queueIDXs.graphics) && (a_queueIDXs.transfer != a_queueIDXs.compute)) || queueCreateInfos.empty())
      {
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = a_queueIDXs.transfer;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &defaultQueuePriority;
        queueCreateInfos.push_back(queueInfo);
      }
    }
    else
    {
      a_queueIDXs.transfer = a_queueIDXs.graphics;
    }


    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.queueCreateInfoCount = queueCreateInfos.size();
    deviceCreateInfo.pEnabledFeatures = &a_deviceFeatures;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(a_extensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = a_extensions.data();

    // deprecated and ignored since Vulkan 1.2
    // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#extendingvulkan-layers-devicelayerdeprecation
    deviceCreateInfo.enabledLayerCount = uint32_t(a_enabledLayers.size());
    deviceCreateInfo.ppEnabledLayerNames = a_enabledLayers.data();

    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{};
    if(pNextFeatures)
    {
      physicalDeviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      physicalDeviceFeatures2.features = a_deviceFeatures;
      physicalDeviceFeatures2.pNext = pNextFeatures;
      deviceCreateInfo.pEnabledFeatures = nullptr;
      deviceCreateInfo.pNext = &physicalDeviceFeatures2;
    }

    VkDevice device;
    VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

    return device;
  }


  uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice)
  {
    VkPhysicalDeviceMemoryProperties memoryProperties;

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
      if ((memoryTypeBits & (1u << i)) && ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
        return i;
    }

    return UINT32_MAX;
  }

  std::vector<std::string> subgroupOperationToString(VkSubgroupFeatureFlags flags)
  {
    std::vector<std::string> res;
    std::vector<std::pair<VkSubgroupFeatureFlagBits, std::string>> flagBits = {
        {VK_SUBGROUP_FEATURE_BASIC_BIT, "VK_SUBGROUP_FEATURE_BASIC_BIT"},
        {VK_SUBGROUP_FEATURE_VOTE_BIT, "VK_SUBGROUP_FEATURE_VOTE_BIT"},
        {VK_SUBGROUP_FEATURE_ARITHMETIC_BIT, "VK_SUBGROUP_FEATURE_ARITHMETIC_BIT"},
        {VK_SUBGROUP_FEATURE_BALLOT_BIT, "VK_SUBGROUP_FEATURE_BALLOT_BIT"},
        {VK_SUBGROUP_FEATURE_SHUFFLE_BIT, "VK_SUBGROUP_FEATURE_SHUFFLE_BIT"},
        {VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT, "VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT"},
        {VK_SUBGROUP_FEATURE_CLUSTERED_BIT, "VK_SUBGROUP_FEATURE_CLUSTERED_BIT"},
        {VK_SUBGROUP_FEATURE_QUAD_BIT, "VK_SUBGROUP_FEATURE_QUAD_BIT"},
        {VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV, "VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV"},
    };
    for(const auto& f : flagBits)
    {
      if(flags & f.first)
        res.emplace_back(f.second);
    }
    return res;
  }

  void executeCommandBufferNow(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device)
  {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &a_cmdBuff;

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));

    VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));

    VK_CHECK_RESULT(vkWaitForFences(a_device, 1, &fence, VK_TRUE, DEFAULT_TIMEOUT));

    vkDestroyFence(a_device, fence, NULL);
  }

  void executeCommandBufferNow(std::vector<VkCommandBuffer> a_cmdBuffers, VkQueue a_queue, VkDevice a_device)
  {
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = a_cmdBuffers.size();
    submitInfo.pCommandBuffers = a_cmdBuffers.data();

    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));

    VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));

    VK_CHECK_RESULT(vkWaitForFences(a_device, 1, &fence, VK_TRUE, DEFAULT_TIMEOUT));

    vkDestroyFence(a_device, fence, NULL);
  }
#ifdef __ANDROID__
  std::vector<uint32_t> readSPVFile(AAssetManager* mgr, const char* filename)
  {
    // Read the file
    assert(mgr);
    AAsset* file = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    auto filesize_padded = getPaddedSize(fileLength, sizeof(uint32_t));

    std::vector<uint32_t> resData(filesize_padded / sizeof(uint32_t), 0);

    auto read_bytes = AAsset_read(file, resData.data(), fileLength);
    if(!read_bytes)
      RUN_TIME_ERROR("[vk_utils::readSPVFile]: AAsset_read error");
    AAsset_close(file);

    return resData;
  }
#else
  std::vector<uint32_t> readSPVFile(const char *filename)
  {
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr)
    {
      std::string errorMsg = std::string("[vk_utils::readSPVFile]: can't open file ") + std::string(filename);
      RUN_TIME_ERROR(errorMsg.c_str());
    }

    fseek(fp, 0, SEEK_END);
    size_t filesize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    auto filesize_padded = getPaddedSize(filesize, sizeof(uint32_t));

    std::vector<uint32_t> resData(filesize_padded / sizeof(uint32_t), 0);

    char *str = (char *)resData.data();
    size_t read_bytes = fread(str, filesize, sizeof(char), fp);
    if(read_bytes != sizeof(char))
      RUN_TIME_ERROR("[vk_utils::readSPVFile]: fread error");
    fclose(fp);

    return resData;
  }
#endif

  VkShaderModule createShaderModule(VkDevice a_device, const std::vector<uint32_t> &code)
  {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();

    VkShaderModule shaderModule;
    VK_CHECK_RESULT(vkCreateShaderModule(a_device, &createInfo, nullptr, &shaderModule));

    return shaderModule;
  }

  VkPipelineShaderStageCreateInfo loadShader(VkDevice a_device, const std::string& fileName, VkShaderStageFlagBits stage,
                                             std::vector<VkShaderModule> &modules)
  {
    VkPipelineShaderStageCreateInfo shaderStage = {};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = stage;

#ifdef __ANDROID__
    shaderStage.module = createShaderModule(a_device, readSPVFile(vk_android::g_pMgr, fileName.c_str()));
#else
    shaderStage.module = createShaderModule(a_device, readSPVFile(fileName.c_str()));
#endif
    shaderStage.pName = "main";
    assert(shaderStage.module != VK_NULL_HANDLE);
    modules.push_back(shaderStage.module);
    return shaderStage;
  }


  VkCommandPool createCommandPool(VkDevice a_device,  uint32_t a_queueIdx, VkCommandPoolCreateFlagBits a_poolFlags)
  {
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = a_poolFlags;
    poolInfo.queueFamilyIndex = a_queueIdx;

    VkCommandPool commandPool;
    VK_CHECK_RESULT(vkCreateCommandPool(a_device, &poolInfo, nullptr, &commandPool));

    return commandPool;
  }

  VkCommandBuffer createCommandBuffer(VkDevice a_device, VkCommandPool a_pool)
  {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = a_pool;
    commandBufferAllocateInfo.level       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &commandBufferAllocateInfo, &cmd));

    return cmd;
  }

  std::vector<VkCommandBuffer> createCommandBuffers(VkDevice a_device, VkCommandPool a_pool, uint32_t a_buffNum)
  {
    std::vector<VkCommandBuffer> commandBuffers(a_buffNum);

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = a_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = a_buffNum;

    VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &allocInfo, commandBuffers.data()));

    return commandBuffers;
  }

  VkRenderPass createDefaultRenderPass(VkDevice a_device, VkFormat a_imageFormat)
  {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = a_imageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    std::array<VkSubpassDependency, 2> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = 0;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = 0;
    dependencies[1].dstAccessMask = 0;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkAttachmentDescription attachments[2] = { colorAttachment, depthAttachment };
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = &attachments[0];
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = dependencies.size();
    renderPassInfo.pDependencies = dependencies.data();

    VkRenderPass res;
    VK_CHECK_RESULT(vkCreateRenderPass(a_device, &renderPassInfo, nullptr, &res));

    return res;
  }

  VkRenderPass createRenderPass(VkDevice a_device, RenderTargetInfo2D a_rtInfo)
  {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = a_rtInfo.format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = a_rtInfo.loadOp;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = a_rtInfo.initialLayout;
    colorAttachment.finalLayout = a_rtInfo.finalLayout;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VkRenderPass res;
    VK_CHECK_RESULT(vkCreateRenderPass(a_device, &renderPassInfo, nullptr, &res))

    return res;
  }

  void setDefaultViewport(VkCommandBuffer a_cmdBuff, float a_width, float a_height)
  {
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width  = a_width;
    viewport.height = a_height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vkCmdSetViewport(a_cmdBuff, 0, 1, &viewport);
  }

  void setDefaultScissor(VkCommandBuffer a_cmdBuff, uint32_t a_width, uint32_t a_height)
  {
    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = { a_width, a_height };

    vkCmdSetScissor(a_cmdBuff, 0, 1, &scissor);
  }

}

