#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>
#include <random>
#include <string>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
#include "vk_utils.h"
#include "vk_buffers.h"
#include "test_class_generated.h"
std::shared_ptr<AppendTest> CreateAppendTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures AppendTest_Generated_ListRequiredDeviceFeatures();

static constexpr uint32_t MAX_SIZE = 1u << 16; // max size of each vector in this test

struct DumpResult
{
  std::vector<uint32_t> a, c, sizes;
  std::vector<uint2>    b;
};

// Calls Dump() via generated 'DumpCmd' in a command buffer with output buffers that are bound only once
// (SetVulkanInOutFor_Dump), so the bindings of member vectors are updated only by CommitAppendedData().
//
struct GPUDumper
{
  GPUDumper(std::shared_ptr<AppendTest> a_pImpl, vk_utils::VulkanContext a_ctx) : m_ctx(a_ctx)
  {
    m_pGen = dynamic_cast<AppendTest_Generated*>(a_pImpl.get());
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    m_bufA   = vk_utils::createBuffer(m_ctx.device, MAX_SIZE*sizeof(uint32_t), usage);
    m_bufB   = vk_utils::createBuffer(m_ctx.device, MAX_SIZE*sizeof(uint2),    usage);
    m_bufC   = vk_utils::createBuffer(m_ctx.device, MAX_SIZE*sizeof(uint32_t), usage);
    m_bufS   = vk_utils::createBuffer(m_ctx.device, 4*sizeof(uint32_t),        usage);
    m_memory = vk_utils::allocateAndBindWithPadding(m_ctx.device, m_ctx.physicalDevice, {m_bufA, m_bufB, m_bufC, m_bufS});
    m_pGen->SetVulkanInOutFor_Dump(m_bufA, 0, m_bufB, 0, m_bufC, 0, m_bufS, 0);
  }

  ~GPUDumper()
  {
    vkDestroyBuffer(m_ctx.device, m_bufA, nullptr);
    vkDestroyBuffer(m_ctx.device, m_bufB, nullptr);
    vkDestroyBuffer(m_ctx.device, m_bufC, nullptr);
    vkDestroyBuffer(m_ctx.device, m_bufS, nullptr);
    vkFreeMemory(m_ctx.device, m_memory, nullptr);
  }

  DumpResult Run(const AppendTest& a_cpu)
  {
    VkCommandBuffer cmdBuff = vk_utils::createCommandBuffer(m_ctx.device, m_ctx.commandPool); // re-recorded after each commit
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmdBuff, &beginInfo);
    m_pGen->DumpCmd(cmdBuff, MAX_SIZE, nullptr, nullptr, nullptr, nullptr);
    vkEndCommandBuffer(cmdBuff);
    vk_utils::executeCommandBufferNow(cmdBuff, m_ctx.computeQueue, m_ctx.device);
    vkFreeCommandBuffers(m_ctx.device, m_ctx.commandPool, 1, &cmdBuff);

    DumpResult res;
    res.sizes.resize(4);
    m_ctx.pCopyHelper->ReadBuffer(m_bufS, 0, res.sizes.data(), 4*sizeof(uint32_t));
    res.a.resize(a_cpu.m_a.size());
    res.b.resize(a_cpu.m_b.size());
    res.c.resize(a_cpu.m_c.size());
    if(!res.a.empty()) m_ctx.pCopyHelper->ReadBuffer(m_bufA, 0, res.a.data(), res.a.size()*sizeof(uint32_t));
    if(!res.b.empty()) m_ctx.pCopyHelper->ReadBuffer(m_bufB, 0, res.b.data(), res.b.size()*sizeof(uint2));
    if(!res.c.empty()) m_ctx.pCopyHelper->ReadBuffer(m_bufC, 0, res.c.data(), res.c.size()*sizeof(uint32_t));
    return res;
  }

  vk_utils::VulkanContext m_ctx;
  AppendTest_Generated*   m_pGen   = nullptr;
  VkBuffer                m_bufA   = VK_NULL_HANDLE, m_bufB = VK_NULL_HANDLE, m_bufC = VK_NULL_HANDLE, m_bufS = VK_NULL_HANDLE;
  VkDeviceMemory          m_memory = VK_NULL_HANDLE;
};

template<typename T>
static uint32_t Hash(const std::vector<T>& a_data) // FNV-1a over bytes
{
  uint32_t hash = 2166136261u;
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(a_data.data());
  for(size_t i = 0; i < a_data.size()*sizeof(T); i++)
    hash = (hash ^ bytes[i]) * 16777619u;
  return hash;
}

static DumpResult RunOnCPU(AppendTest& a_impl)
{
  DumpResult res;
  res.a.resize(MAX_SIZE); res.b.resize(MAX_SIZE); res.c.resize(MAX_SIZE); res.sizes.resize(4);
  a_impl.Dump(MAX_SIZE, res.a.data(), res.b.data(), res.c.data(), res.sizes.data());
  res.a.resize(a_impl.m_a.size()); res.b.resize(a_impl.m_b.size()); res.c.resize(a_impl.m_c.size());
  return res;
}

static bool Check(const AppendTest& a_impl, const DumpResult& a_res, const std::string& a_step)
{
  bool ok = (a_res.sizes[0] == a_impl.m_a.size()) && (a_res.sizes[1] == a_impl.m_b.size()) && (a_res.sizes[2] == a_impl.m_c.size()) && (a_res.sizes[3] == 12345u);
  for(size_t i = 0; ok && i < a_impl.m_a.size(); i++)
    ok = (a_res.a[i] == a_impl.m_a[i]*3u + 1u);
  for(size_t i = 0; ok && i < a_impl.m_b.size(); i++)
    ok = (a_res.b[i].x == a_impl.m_b[i].x + 7u) && (a_res.b[i].y == a_impl.m_b[i].y*5u);
  for(size_t i = 0; ok && i < a_impl.m_c.size(); i++)
    ok = (a_res.c[i] == (a_impl.m_c[i] ^ 0x5A5A5A5Au));
  if(!ok)
    std::cout << "[test_064]: FAILED at step '" << a_step << "', sizes on GPU = (" << a_res.sizes[0] << ", " << a_res.sizes[1] << ", " << a_res.sizes[2]
              << "), on CPU = (" << a_impl.m_a.size() << ", " << a_impl.m_b.size() << ", " << a_impl.m_c.size() << ")" << std::endl;
  return ok;
}

static void Append(AppendTest& a_impl, int a_vecId, size_t a_count, std::mt19937& a_gen)
{
  for(size_t i = 0; i < a_count; i++)
  {
    const uint32_t val = uint32_t(a_gen());
    if(a_vecId == 0 && a_impl.m_a.size() < MAX_SIZE)
      a_impl.m_a.push_back(val);
    else if(a_vecId == 1 && a_impl.m_b.size() < MAX_SIZE)
      a_impl.m_b.push_back(uint2(val, val >> 7));
    else if(a_vecId == 2 && a_impl.m_c.size() < MAX_SIZE)
      a_impl.m_c.push_back(val);
  }
}

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;

  std::shared_ptr<AppendTest> pImpl = nullptr;
  vk_utils::VulkanContext     ctx;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = AppendTest_Generated_ListRequiredDeviceFeatures();
    ctx           = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateAppendTest_Generated(ctx, MAX_SIZE);
  }
  else
    pImpl = std::make_shared<AppendTest>();

  std::string backendName = onGPU ? "gpu" : "cpu";
  std::mt19937 gen(12345);
  bool allOk = true;
  uint32_t cpuCapacityChanges = 0;
  {
    std::unique_ptr<GPUDumper> pDumper = nullptr;
    auto Step = [&](const std::string& a_name)
    {
      DumpResult res = onGPU ? pDumper->Run(*pImpl) : RunOnCPU(*pImpl);
      allOk = Check(*pImpl, res, a_name) && allOk;
      JSONLog::write((a_name + "_sizes").c_str(), res.sizes);  // content is written as a checksum to keep the file small
      JSONLog::write((a_name + "_hash").c_str(),  std::vector<uint32_t>{Hash(res.a), Hash(res.b), Hash(res.c)});
    };
    auto Realloc = [&](const char* a_group) -> uint32_t
    {
      auto pGen = dynamic_cast<AppendTest_Generated*>(pImpl.get());
      return (pGen == nullptr) ? 0 : pGen->GetAppendGroupReallocCount(a_group);
    };
    auto Expect = [&](bool a_cond, const char* a_msg)
    {
      if(onGPU && !a_cond)
      {
        std::cout << "[test_064]: FAILED, " << a_msg << std::endl;
        allOk = false;
      }
    };

    // (1) initial data and full commit
    pImpl->m_a.reserve(128); Append(*pImpl, 0, 100, gen);
    pImpl->m_b.reserve(64);  Append(*pImpl, 1, 40,  gen);
    pImpl->m_c.reserve(16);  Append(*pImpl, 2, 10,  gen);
    pImpl->CommitDeviceData();
    if(onGPU)
      pDumper = std::make_unique<GPUDumper>(pImpl, ctx);
    Step("s1_commit");

    // (2) append within capacity: no re-allocation, only appended elements are copied
    Append(*pImpl, 0, 20, gen);
    Append(*pImpl, 2, 5,  gen);
    pImpl->CommitAppendedData();
    Step("s2_append");
    Expect(Realloc("geom") == 0 && Realloc("otherGroup") == 0, "step 2 must not re-allocate");

    // (3) 'm_b' outgrows its buffer: group 'geom' is re-allocated, 'm_a' must be copied again
    Append(*pImpl, 1, 160, gen);
    pImpl->CommitAppendedData();
    Step("s3_realloc_geom");
    Expect(Realloc("geom") == 1 && Realloc("otherGroup") == 0, "step 3 must re-allocate 'geom' only");

    // (4) 'm_c' outgrows its buffer: 'otherGroup' is re-allocated
    Append(*pImpl, 2, 500, gen);
    pImpl->CommitAppendedData();
    Step("s4_realloc_other");
    Expect(Realloc("geom") == 1 && Realloc("otherGroup") == 1, "step 4 must re-allocate 'otherGroup' only");

    // (5) shrink: only sizes change
    pImpl->m_a.resize(50);
    pImpl->CommitAppendedData();
    Step("s5_shrink");
    Expect(Realloc("geom") == 1 && Realloc("otherGroup") == 1, "step 5 must not re-allocate");

    // (6) many random appends (and rare shrinks) in both groups
    for(int iter = 0; iter < 300; iter++)
    {
      const int vecId     = int(gen() % 3);
      const size_t oldCap = (vecId == 0) ? pImpl->m_a.capacity() : (vecId == 1) ? pImpl->m_b.capacity() : pImpl->m_c.capacity();
      if(gen() % 16 == 0)
      {
        if(vecId == 0) pImpl->m_a.resize(pImpl->m_a.size()/2);
        if(vecId == 1) pImpl->m_b.resize(pImpl->m_b.size()/2);
        if(vecId == 2) pImpl->m_c.resize(pImpl->m_c.size()/2);
      }
      else
        Append(*pImpl, vecId, gen() % 600, gen);
      const size_t newCap = (vecId == 0) ? pImpl->m_a.capacity() : (vecId == 1) ? pImpl->m_b.capacity() : pImpl->m_c.capacity();
      if(newCap != oldCap)
        cpuCapacityChanges++;
      pImpl->CommitAppendedData();
      Step("s6_random_" + std::to_string(iter));
    }

    // (7) grow factor 1: GPU capacity becomes exactly the capacity on CPU; GPU capacity is at most 2*capacity()
    //     with grow factor 2, so reserve(4*capacity()) surely overflows the GPU buffer (the same data on CPU and GPU)
    {
      auto pGen = dynamic_cast<AppendTest_Generated*>(pImpl.get());
      if(pGen != nullptr)
        pGen->SetAppendGrowFactor(1.0f);
      pImpl->m_c.reserve(4*pImpl->m_c.capacity() + 1000);
      Append(*pImpl, 2, 10, gen);
      pImpl->CommitAppendedData();
      if(pGen != nullptr)
      {
        Expect(pGen->GetAppendGpuCapacity("m_c") == pImpl->m_c.capacity(), "step 7: GPU capacity must be equal to capacity() with grow factor 1");
        pGen->SetAppendGrowFactor(2.0f);
      }
    }
    Step("s7_grow_factor");

    // (8) full commit and appends after it; CommitDeviceData() re-creates all device data, so bindings
    //     must be set again (unlike CommitAppendedData(), which updates them itself)
    pImpl->CommitDeviceData();
    if(onGPU)
      pDumper->m_pGen->SetVulkanInOutFor_Dump(pDumper->m_bufA, 0, pDumper->m_bufB, 0, pDumper->m_bufC, 0, pDumper->m_bufS, 0);
    Append(*pImpl, 0, 3000, gen);
    pImpl->CommitAppendedData();
    Step("s8_after_full_commit");

    // (9) usual (non Cmd) call of the control function after all re-allocations
    {
      DumpResult res;
      res.a.resize(MAX_SIZE); res.b.resize(MAX_SIZE); res.c.resize(MAX_SIZE); res.sizes.resize(4);
      pImpl->Dump(MAX_SIZE, res.a.data(), res.b.data(), res.c.data(), res.sizes.data());
      res.a.resize(pImpl->m_a.size()); res.b.resize(pImpl->m_b.size()); res.c.resize(pImpl->m_c.size());
      allOk = Check(*pImpl, res, "s9_usual_call") && allOk;
    }
    pDumper = nullptr;
  }

  if(onGPU)
    std::cout << "[test_064]: re-allocations: geom = " << dynamic_cast<AppendTest_Generated*>(pImpl.get())->GetAppendGroupReallocCount("geom")
              << ", otherGroup = " << dynamic_cast<AppendTest_Generated*>(pImpl.get())->GetAppendGroupReallocCount("otherGroup")
              << ", capacity changes on CPU in random steps = " << cpuCapacityChanges << std::endl;
  std::cout << (allOk ? "[test_064]: PASSED" : "[test_064]: FAILED") << std::endl;

  JSONLog::write("passed", allOk ? 1 : 0);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  ctx   = vk_utils::VulkanContext{}; // release copy helper before the device is destroyed
  if(onGPU)
    vk_utils::globalContextDestroy();
  return allOk ? 0 : 1;
}
