#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include <vector>
#include <iostream>
#include <memory>
#include <chrono>



#include "vk_utils.h"
#include "vk_pipeline.h"
#include "vk_copy.h"
#include "vk_buffers.h"

#include "test_class_generated.h"
#include "test_class_gpu.h"

const int GRID_SIZE = 2;

double randfrom(double min, double max) {
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void save_image(int N, const std::string &image_name, std::vector<float> density) {
    std::vector<unsigned char> image;
    image.resize(N * N * 4 * GRID_SIZE * GRID_SIZE);

    float d_min = *std::min_element(std::begin(density), std::end(density));
    float d_max = *std::max_element(std::begin(density), std::end(density));
    int grid_size = GRID_SIZE;
    for (int i = 0; i < N * N; ++i) {
        float d = (density[i] - d_min) / (d_max - d_min);
        for (int j = 0; j < grid_size; ++j) {
            for (int k = 0; k < grid_size; ++k) {
                int indx = 4 * (i % N * grid_size + (i / N) * grid_size * grid_size * N + k + j * N * grid_size);
                image[indx] = 0;
                image[indx + 1] = (unsigned char)155.0f * d;
                image[indx + 2] = (unsigned char) 255.0f * d;
                image[indx + 3] = 255;
            }
        }

    }

    stbi_write_bmp(image_name.c_str(), N * grid_size, N * grid_size, 4, &image[0]);
}

std::vector<float> solve_gpu(int N, const std::vector<float>& density, const std::vector<float>& vx, const std::vector<float>& vy) {
    std::vector<float> outDens(N * N);

    // (1) init vulkan
    //
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;

#ifndef NDEBUG
    bool enableValidationLayers = true;
#else
    bool enableValidationLayers = false;
#endif

    std::vector<const char *> enabledLayers;
    std::vector<const char *> extensions;
    enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
    VK_CHECK_RESULT(volkInitialize());
    instance = vk_utils::createInstance(enableValidationLayers, enabledLayers, extensions);
    volkLoadInstance(instance);

    physicalDevice = vk_utils::findPhysicalDevice(instance, true, 0);
    // query for shaderInt8
    //

    std::vector<const char *> validationLayers, deviceExtensions;
    VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
    vk_utils::QueueFID_T fIDs = {};

    device = vk_utils::createLogicalDevice(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures,
                                           fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT,
                                           nullptr);
    volkLoadDevice(device);

    commandPool = vk_utils::createCommandPool(device, fIDs.compute, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

    // (2) initialize vulkan helpers
    //
    VkQueue computeQueue, transferQueue;
    {
        vkGetDeviceQueue(device, fIDs.compute, 0, &computeQueue);
        vkGetDeviceQueue(device, fIDs.transfer, 0, &transferQueue);
    }

    auto pCopyHelper = std::make_shared<vk_utils::SimpleCopyHelper>(physicalDevice, device, transferQueue,
                                                                    fIDs.transfer,
                                                                    8 * 1024 * 1024);

    auto pGPUImpl = std::make_shared<Solver_Generated>();          // !!! USING GENERATED CODE !!!
    pGPUImpl->setParameters(N, density, vx, vy, 0.033, 0, 0);
    pGPUImpl->InitVulkanObjects(device, physicalDevice, 1); // !!! USING GENERATED CODE !!!

    pGPUImpl->InitMemberBuffers();                                      // !!! USING GENERATED CODE !!!
    pGPUImpl->UpdateAll(pCopyHelper);                                   // !!! USING GENERATED CODE !!!

    // (3) Create buffer
    //
    VkBuffer outBuffer = vk_utils::createBuffer(device, N * N * sizeof(float),
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    VkDeviceMemory bufferMem = vk_utils::allocateAndBindWithPadding(device, physicalDevice, {outBuffer});
    pGPUImpl->SetVulkanInOutFor_perform(outBuffer, 0);
   
    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(device, commandPool);
    
    // all iterations at once
    //
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    for (int i = 0; i < 50; ++i)
       pGPUImpl->performCmd(commandBuffer, nullptr);
    vkEndCommandBuffer(commandBuffer);
    
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;
    
    std::cout << ms << " ms for command buffer execution " << std::endl;
    pCopyHelper->ReadBuffer(outBuffer, 0, outDens.data(), outDens.size() * sizeof(float));

    //// iter by iter 
    //
    /*
    for (int i = 0; i < 50; ++i) {
        VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(device, commandPool);
    
        VkCommandBufferBeginInfo beginCommandBufferInfo = {};
        beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
        // vkCmdFillBuffer(commandBuffer, outBuffer, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
        pGPUImpl->performCmd(commandBuffer, nullptr);         // !!! USING GENERATED CODE !!!
        vkEndCommandBuffer(commandBuffer);
    
        auto start = std::chrono::high_resolution_clock::now();
        vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
        auto stop = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;
        std::cout << ms << " ms for command buffer execution " << std::endl;

        pCopyHelper->ReadBuffer(outBuffer, 0, outDens.data(), outDens.size() * sizeof(float));
//        pCopyHelper->ReadBuffer(pGPUImpl.get()->m_vdata.stagingBuff, 0, outDens.data(), outDens.size() * sizeof(float));
        save_image(N, "images_gpu/" + std::to_string(i) + ".jpeg", outDens);
        std::cout << std::endl;
    }*/
    
    // (6) destroy and free resources before exit
    //
    pCopyHelper = nullptr;
    pGPUImpl = nullptr;                                                       // !!! USING GENERATED CODE !!!

    vkDestroyBuffer(device, outBuffer, nullptr);
    vkFreeMemory(device, bufferMem, nullptr);

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return outDens;
}