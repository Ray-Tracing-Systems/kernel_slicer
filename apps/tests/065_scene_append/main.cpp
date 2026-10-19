#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>
#include <random>
#include <string>
#include <cmath>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
#include "vk_utils.h"
#include "vk_buffers.h"
#include "BVH2CommonRT.h"
#include "test_class_generated.h"
std::shared_ptr<SceneAppend> CreateSceneAppend_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures SceneAppend_Generated_ListRequiredDeviceFeatures();

static constexpr uint32_t WIDTH  = 512;
static constexpr uint32_t HEIGHT = 512;

// software ray tracing: TLAS in 'm_allNodes' is rebuilt by CommitScene(), BVH2CommonRT reports the changed range via this hook
//
static SceneAppend_Generated* g_pGenerated = nullptr;
static void InvalidateNodes(size_t a_first, size_t a_size) 
{ 
  if(g_pGenerated != nullptr) 
    g_pGenerated->InvalidateAppendedData("m_pAccelStruct_m_allNodes", a_first); 
}

struct MeshData
{
  std::vector<float4>   verts;
  std::vector<uint32_t> indices;
};

static MeshData MakeMesh(std::mt19937& a_gen, uint32_t a_trisNum)
{
  std::uniform_real_distribution<float> pos(-1.0f, 1.0f);
  MeshData mesh;
  for(uint32_t i = 0; i < a_trisNum; i++)
  {
    const float4 center = float4(pos(a_gen), pos(a_gen), pos(a_gen), 1.0f);
    for(int k = 0; k < 3; k++)
    {
      mesh.verts.push_back(center + float4(0.3f*pos(a_gen), 0.3f*pos(a_gen), 0.3f*pos(a_gen), 0.0f));
      mesh.indices.push_back(uint32_t(mesh.verts.size() - 1));
    }
  }
  return mesh;
}

static float4x4 MakeMatrix(std::mt19937& a_gen)
{
  std::uniform_real_distribution<float> tr(-8.0f, 8.0f);
  std::uniform_real_distribution<float> angle(0.0f, 6.28f);
  std::uniform_real_distribution<float> scale(0.5f, 2.0f);
  return LiteMath::translate4x4(LiteMath::float3(tr(a_gen), tr(a_gen), tr(a_gen))) * LiteMath::rotate4x4Z(angle(a_gen)) * 
         LiteMath::rotate4x4X(angle(a_gen)) * LiteMath::scale4x4(LiteMath::float3(scale(a_gen)));
}

// Calls Render() via generated 'RenderCmd' with an output buffer bound only once, so bindings of
// re-allocated member buffers and of re-created TLAS are updated only by CommitAppendedData()
//
struct GPURenderer
{
  GPURenderer(SceneAppend_Generated* a_pGen, vk_utils::VulkanContext a_ctx) : m_pGen(a_pGen), m_ctx(a_ctx)
  {
    m_buffer = vk_utils::createBuffer(m_ctx.device, WIDTH*HEIGHT*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_memory = vk_utils::allocateAndBindWithPadding(m_ctx.device, m_ctx.physicalDevice, {m_buffer});
    m_pGen->SetVulkanInOutFor_Render(m_buffer, 0);
  }
  ~GPURenderer()
  {
    vkDestroyBuffer(m_ctx.device, m_buffer, nullptr);
    vkFreeMemory(m_ctx.device, m_memory, nullptr);
  }
  std::vector<uint32_t> Render()
  {
    VkCommandBuffer cmdBuff = vk_utils::createCommandBuffer(m_ctx.device, m_ctx.commandPool);
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmdBuff, &beginInfo);
    m_pGen->RenderCmd(cmdBuff, WIDTH, HEIGHT, nullptr);
    vkEndCommandBuffer(cmdBuff);
    vk_utils::executeCommandBufferNow(cmdBuff, m_ctx.computeQueue, m_ctx.device);
    vkFreeCommandBuffers(m_ctx.device, m_ctx.commandPool, 1, &cmdBuff);
    std::vector<uint32_t> image(WIDTH*HEIGHT);
    m_ctx.pCopyHelper->ReadBuffer(m_buffer, 0, image.data(), image.size()*sizeof(uint32_t));
    return image;
  }
  SceneAppend_Generated*  m_pGen   = nullptr;
  vk_utils::VulkanContext m_ctx;
  VkBuffer                m_buffer = VK_NULL_HANDLE;
  VkDeviceMemory          m_memory = VK_NULL_HANDLE;
};

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

  std::shared_ptr<SceneAppend> pRef  = std::make_shared<SceneAppend>(); // CPU reference, gets the same operations
  std::shared_ptr<SceneAppend> pImpl = nullptr;
  vk_utils::VulkanContext      ctx;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = SceneAppend_Generated_ListRequiredDeviceFeatures();
    ctx           = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateSceneAppend_Generated(ctx, WIDTH*HEIGHT);
    g_pGenerated  = dynamic_cast<SceneAppend_Generated*>(pImpl.get());
    auto pSoftwareImpl = dynamic_cast<BVH2CommonRT*>(pImpl->m_pAccelStruct->UnderlyingImpl(0));
    if(pSoftwareImpl != nullptr && pImpl->m_pAccelStruct->UnderlyingImpl(1) == pSoftwareImpl) // software ray tracing on GPU
      pSoftwareImpl->m_pUpdate_m_allNodes = &InvalidateNodes;
  }
  else
    pImpl = std::make_shared<SceneAppend>();

  const std::string backendName = onGPU ? "gpu" : "cpu";
  std::mt19937 gen(777);
  std::vector<uint32_t> instGeom;
  std::vector<float4x4> instMatr;
  std::vector<uint32_t> geomIds;
  bool   allOk      = true;
  size_t totalTris  = 0;

  auto AddMeshes = [&](int a_count, uint32_t a_maxTris)
  {
    for(int i = 0; i < a_count; i++)
    {
      const MeshData mesh = MakeMesh(gen, 1 + gen() % a_maxTris);
      const uint32_t id1  = pRef->AddMesh(mesh.verts, mesh.indices);
      const uint32_t id2  = pImpl->AddMesh(mesh.verts, mesh.indices);
      if(id1 != id2)
      {
        std::cout << "[test_065]: FAILED, different geometry id on CPU and GPU: " << id1 << ", " << id2 << std::endl;
        allOk = false;
      }
      geomIds.push_back(id1);
      totalTris += mesh.indices.size()/3;
    }
  };
  auto AddInstances = [&](int a_count, bool a_onlyNewest)
  {
    for(int i = 0; i < a_count; i++)
    {
      const uint32_t geomId = a_onlyNewest ? geomIds.back() : geomIds[gen() % geomIds.size()];
      instGeom.push_back(geomId);
      instMatr.push_back(MakeMatrix(gen));
    }
  };

  std::unique_ptr<GPURenderer> pRenderer = nullptr;
  auto Step = [&](const std::string& a_name)
  {
    pRef->SetInstances(instGeom, instMatr);
    pImpl->SetInstances(instGeom, instMatr);
    if(pRenderer == nullptr)
    {
      pImpl->CommitDeviceData();
      if(onGPU)
        pRenderer = std::make_unique<GPURenderer>(g_pGenerated, ctx);
    }
    else
      pImpl->CommitAppendedData();

    std::vector<uint32_t> imgRef(WIDTH*HEIGHT), img(WIDTH*HEIGHT);
    pRef->Render(WIDTH, HEIGHT, imgRef.data());
    if(onGPU)
      img = pRenderer->Render();
    else
      pImpl->Render(WIDTH, HEIGHT, img.data());

    size_t diff = 0, hits = 0;
    for(size_t i = 0; i < img.size(); i++)
    {
      if(img[i] != imgRef[i])
        diff++;
      if(imgRef[i] != 0xFFFFFFFFu)
        hits++;
    }
    const double diffPercent = 100.0*double(diff)/double(img.size()); // hardware traversal may resolve ties on triangle edges differently
    const bool   ok          = (diffPercent < 0.5) && (hits > 0);
    allOk = allOk && ok;
    if(!ok)
      std::cout << "[test_065]: FAILED at step '" << a_name << "': " << diff << " pixels differ (" << diffPercent << "%), hits = " << hits << std::endl;
    JSONLog::write((a_name + "_hits").c_str(), int(hits));
  };

  // (1) initial scene, full commit
  AddMeshes(3, 200);
  AddInstances(5, false);
  Step("s0_initial");

  // (2) new geometry and instances after CommitDeviceData(), then only instance changes, many times
  for(int iter = 0; iter < 40; iter++)
  {
    const int action = int(gen() % 4);
    if(action == 0)                        // only move instances
    {
      for(auto& m : instMatr)
        m = MakeMatrix(gen);
    }
    else if(action == 1)                   // add instances of existing geometry
      AddInstances(1 + int(gen() % 5), false);
    else                                   // add new geometry (BLAS) and its instances
    {
      AddMeshes(1 + int(gen() % 3), (iter % 10 == 9) ? 20000 : 1000);
      AddInstances(1 + int(gen() % 3), true);
    }
    Step("s1_iter_" + std::to_string(iter));
  }

  // (3) many instances at once (TLAS grows a lot)
  AddInstances(300, false);
  Step("s2_many_instances");

  if(g_pGenerated != nullptr)
    std::cout << "[test_065]: geometries = " << geomIds.size() << ", triangles = " << totalTris << ", instances = " << instGeom.size() 
              << ", re-allocations of BVH data = " << g_pGenerated->GetAppendGroupReallocCount("bvh") << std::endl;
  std::cout << (allOk ? "[test_065]: PASSED" : "[test_065]: FAILED") << std::endl;

  JSONLog::write("passed", allOk ? 1 : 0);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pRenderer    = nullptr;
  g_pGenerated = nullptr;
  pImpl        = nullptr;
  ctx          = vk_utils::VulkanContext{};
  if(onGPU)
    vk_utils::globalContextDestroy();
  return allOk ? 0 : 1;
}
