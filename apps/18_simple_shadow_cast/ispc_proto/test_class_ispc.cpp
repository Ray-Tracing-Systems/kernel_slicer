// Hand-written prototype of what kslicer ISPC back end should generate for 'TestClass' (IPV, ray tracing via Embree packets).
//
#include <iostream>
#include <memory>
#include <cstring>
#include <cstdlib>

#include "test_class.h"
#include "test_class_kernels.h"

#ifdef USE_EMBREE4
#include "EmbreeRT4.h"
using EmbreeImpl = EmbreeRT4;
#else
#include "EmbreeRT.h"
using EmbreeImpl = EmbreeRT;
#endif

class TestClass_ISPC : public TestClass
{
public:

  TestClass_ISPC(int a_maxThreads) : TestClass(a_maxThreads) {}
  ~TestClass_ISPC(){}

  void CommitDeviceData() override
  {
    UpdatePlainMembers();
  }

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY) override;
  void CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color) override;

protected:

  virtual void UpdatePlainMembers();

  void kernel2D_PackXY(uint tidX, uint tidY, uint* out_pakedXY) override
  {
    ispc::kernel2D_PackXY_ISPC(tidX, tidY, out_pakedXY, &m_uboData);
  }

  void kernel1D_CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color) override
  {
    // get Embree scene from 'm_pAccelStruct' in the same way as Vulkan back end gets VulkanRTX implementation
    //
    auto pEmbree = dynamic_cast<EmbreeImpl*>(m_pAccelStruct->UnderlyingImpl(0));
    if(pEmbree == nullptr)
    {
      std::cout << "[TestClass_ISPC]: m_pAccelStruct is not an Embree implementation, can't trace rays with ISPC" << std::endl;
      exit(0);
    }
    ispc::kernel1D_CastSingleRay_ISPC(tid, in_pakedXY, out_color, m_materialIds.data(), (const ispc::float4*)m_materials.data(),
                                      (ispc::RandomGen*)m_randomGens.data(), pEmbree->GetEmbreeScene(), pEmbree->GetGeomIdByInstId(), &m_uboData);
  }

  ispc::TestClass_ISPC_UBO_Data m_uboData;
};

static_assert(sizeof(ispc::LightGeom) == sizeof(LightGeom),            "LightGeom layout mismatch");
static_assert(sizeof(ispc::float4x4T) == sizeof(LiteMath::float4x4),   "float4x4 layout mismatch");
static_assert(sizeof(ispc::RandomGen) == sizeof(RandomGen),            "RandomGen layout mismatch");

void TestClass_ISPC::UpdatePlainMembers()
{
  memcpy(&m_uboData.m_worldViewProjInv, &m_worldViewProjInv, sizeof(m_worldViewProjInv));
  memcpy(&m_uboData.m_lightGeom, &m_lightGeom, sizeof(m_lightGeom));
  const float camPos[4] = {m_camPos.x, m_camPos.y, m_camPos.z, 0.0f};          // LiteMath::float3 is 12 bytes, ISPC float<3> is 16 bytes
  memcpy(&m_uboData.m_camPos, camPos, sizeof(camPos));
  m_uboData.m_emissiveMaterialId   = m_emissiveMaterialId;
  m_uboData.m_materialIds_size     = uint32_t(m_materialIds.size());
  m_uboData.m_materialIds_capacity = uint32_t(m_materialIds.capacity());
  m_uboData.m_materials_size       = uint32_t(m_materials.size());
  m_uboData.m_materials_capacity   = uint32_t(m_materials.capacity());
  m_uboData.m_randomGens_size      = uint32_t(m_randomGens.size());
  m_uboData.m_randomGens_capacity  = uint32_t(m_randomGens.capacity());
}

void TestClass_ISPC::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  UpdatePlainMembers();
  TestClass::PackXY(tidX, tidY, out_pakedXY);
}

void TestClass_ISPC::CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color)
{
  UpdatePlainMembers();
  TestClass::CastSingleRay(tid, in_pakedXY, out_color); // measures time itself, see TestClass::GetExecutionTime
}

std::shared_ptr<TestClass> CreateTestClass_ISPC(int a_maxThreads)
{
  return std::make_shared<TestClass_ISPC>(a_maxThreads);
}
