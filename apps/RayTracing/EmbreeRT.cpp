#include <iostream>
#include <vector>
#include <unordered_map>
#include <cassert>

#include "CrossRT.h"
#include "embree3/rtcore.h"

class EmbreeRT : public ISceneObject
{
public:
  EmbreeRT();
  ~EmbreeRT();
  void ClearGeom() override;
  
  uint32_t AddGeom_Triangles4f(const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber) override;
  void     UpdateGeom_Triangles4f(uint32_t a_geomId, const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber) override;

  void ClearScene() override; 
  void CommitScene  () override; 
  
  uint32_t AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix) override;
  void     UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix) override;

  CRT_Hit  RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override;
  bool     RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override;

protected:
  RTCDevice m_device = nullptr;
  RTCScene  m_scene  = nullptr;

  std::vector<RTCScene>    m_blas;
  std::vector<RTCGeometry> m_inst;
  std::vector<uint32_t>    m_geomIdByInstId;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void error_handler(void* userPtr, const RTCError code, const char* str)
{
  if (code == RTC_ERROR_NONE)
    return;
  
  std::cout << ("Embree: ");
  switch (code) {
  case RTC_ERROR_UNKNOWN          : std::cout << "RTC_ERROR_UNKNOWN"; break;
  case RTC_ERROR_INVALID_ARGUMENT : std::cout << "RTC_ERROR_INVALID_ARGUMENT"; break;
  case RTC_ERROR_INVALID_OPERATION: std::cout << "RTC_ERROR_INVALID_OPERATION"; break;
  case RTC_ERROR_OUT_OF_MEMORY    : std::cout << "RTC_ERROR_OUT_OF_MEMORY"; break;
  case RTC_ERROR_UNSUPPORTED_CPU  : std::cout << "RTC_ERROR_UNSUPPORTED_CPU"; break;
  case RTC_ERROR_CANCELLED        : std::cout << "RTC_ERROR_CANCELLED"; break;
  default                         : std::cout << "invalid error code"; break;
  }
  if (str) {
    std::cout << " (";
    while (*str) std::cout << (*str++);
    std::cout << ")\n";
  }
  exit(1);
}


EmbreeRT::EmbreeRT()
{
  m_device = rtcNewDevice("isa=avx2");
  m_scene  = nullptr;
  
  rtcSetDeviceErrorFunction(m_device, error_handler, nullptr);
  m_blas.reserve(1024);
  m_inst.reserve(2048);
  m_geomIdByInstId.reserve(m_inst.capacity());
}

EmbreeRT::~EmbreeRT()
{
  rtcReleaseScene(m_scene);
  rtcReleaseDevice(m_device);
}

void EmbreeRT::ClearGeom()
{
  for(auto& scn : m_blas)
    rtcReleaseScene(scn);
  
  if(m_scene != nullptr)
    rtcReleaseScene(m_scene);
  m_scene = rtcNewScene(m_device);
  rtcSetSceneBuildQuality(m_scene, RTC_BUILD_QUALITY_HIGH);

  m_blas.resize(0);
  m_inst.resize(0);
  m_geomIdByInstId.resize(0);
}
  
uint32_t EmbreeRT::AddGeom_Triangles4f(const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber)
{ 
  if(a_vpos4f == nullptr)
  {
    std::cout << "EmbreeRT::AddGeom_Triangles4f, nullptr input: a_vpos4f" << std::endl;
    return uint32_t(-1);
  }

  if(a_triIndices == nullptr)
  {
    std::cout << "EmbreeRT::AddGeom_Triangles4f, nullptr input: a_triIndices" << std::endl;
    return uint32_t(-1);
  }

  RTCGeometry geom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_TRIANGLE);

  float* vertices   = (float*)    rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 4*sizeof(float),    a_vertNumber);
  unsigned* indices = (unsigned*) rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX,  0,  RTC_FORMAT_UINT3, 3*sizeof(unsigned), a_indNumber/3);

  memcpy(vertices, a_vpos4f, a_vertNumber*4*sizeof(float));
  memcpy(indices,  a_triIndices, a_indNumber*sizeof(unsigned));

  rtcCommitGeometry(geom);

  // attach 'geom' to 'meshScene' and then remember 'meshScene' in 'm_blas'
  //
  auto meshScene = rtcNewScene(m_device);
  rtcSetSceneBuildQuality(meshScene, RTC_BUILD_QUALITY_HIGH);
  
  uint32_t geomId = rtcAttachGeometry(meshScene, geom); 
  rtcReleaseGeometry(geom);
  m_blas.push_back(meshScene);

  rtcCommitScene(meshScene);
  return uint32_t(m_blas.size()-1);
}

void EmbreeRT::UpdateGeom_Triangles4f(uint32_t a_geomId, const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber)
{
  std::cout << "EmbreeRT::UpdateGeom_Triangles4f is not implemented yet!" << std::endl;  
}

void EmbreeRT::ClearScene()
{
  m_inst.resize(0);
  if(m_scene != nullptr)
    rtcReleaseScene(m_scene);
  m_scene = rtcNewScene(m_device);
  rtcSetSceneBuildQuality(m_scene, RTC_BUILD_QUALITY_HIGH);
} 

uint32_t EmbreeRT::AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix)
{
  if(a_geomId >= m_blas.size())
    return uint32_t(-1);

  RTCGeometry instanceGeom = rtcNewGeometry(m_device, RTC_GEOMETRY_TYPE_INSTANCE);
  rtcSetGeometryInstancedScene(instanceGeom, m_blas[a_geomId]);                   // say that 'instanceGeom' is an instance of 'm_blas[a_geomId]'
  //rtcSetGeometryTimeStepCount(instanceGeom, 1);                                   // don't know wtf is that
  
  rtcAttachGeometry(m_scene,instanceGeom);                                        // attach our instance to global scene 
  rtcReleaseGeometry(instanceGeom);
  
  // update instance matrix
  //
  rtcSetGeometryTransform(instanceGeom, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, (const float*)&a_matrix);
  rtcCommitGeometry(instanceGeom);
  
  m_inst.push_back(instanceGeom);
  m_geomIdByInstId.push_back(a_geomId);
  return uint32_t(m_inst.size()-1);
}

void EmbreeRT::CommitScene()
{
  rtcCommitScene(m_scene);
}  


void  EmbreeRT::UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix)
{
  if(a_instanceId >= m_inst.size())
    return;

  rtcSetGeometryTransform(m_inst[a_instanceId], 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, (const float*)&a_matrix);
  rtcCommitGeometry(m_inst[a_instanceId]);
}

CRT_Hit  EmbreeRT::RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar)
{    
  // The intersect context can be used to set intersection
  // filters or flags, and it also contains the instance ID stack
  // used in multi-level instancing.
  // 
  struct RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  // The ray hit structure holds both the ray and the hit.
  // The user must initialize it properly -- see API documentation
  // for rtcIntersect1() for details.
  //  
  struct RTCRayHit rayhit;
  rayhit.ray.org_x = posAndNear.x;
  rayhit.ray.org_y = posAndNear.y;
  rayhit.ray.org_z = posAndNear.z;
  rayhit.ray.tnear = posAndNear.w;

  rayhit.ray.dir_x = dirAndFar.x;
  rayhit.ray.dir_y = dirAndFar.y;
  rayhit.ray.dir_z = dirAndFar.z;
  rayhit.ray.tfar  = dirAndFar.w; // std::numeric_limits<float>::infinity();
  
  rayhit.ray.mask   = -1;
  rayhit.ray.flags  = 0;
  rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  // There are multiple variants of rtcIntersect. This one intersects a single ray with the scene.
  // 
  rtcIntersect1(m_scene, &context, &rayhit);

  CRT_Hit result;
  if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
  {
    result.t      = rayhit.ray.tfar;
    result.geomId = m_geomIdByInstId[rayhit.hit.instID[0]];
    result.instId = rayhit.hit.instID[0];
    result.primId = rayhit.hit.primID;
    result.coords[1] = rayhit.hit.u;
    result.coords[0] = rayhit.hit.v;
    result.coords[2] = 1.0f - rayhit.hit.v - rayhit.hit.u;
  }
  else
  {
    result.t      = rayhit.ray.tfar;
    result.geomId = uint32_t(-1);
    result.instId = uint32_t(-1);
    result.primId = uint32_t(-1);
  }

  return result;
}

bool EmbreeRT::RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar)
{
  // The intersect context can be used to set intersection
  // filters or flags, and it also contains the instance ID stack
  // used in multi-level instancing.
  // 
  struct RTCIntersectContext context;
  rtcInitIntersectContext(&context);

  // The ray hit structure holds both the ray and the hit.
  // The user must initialize it properly -- see API documentation
  // for rtcIntersect1() for details.
  //  
  struct RTCRay ray;
  ray.org_x = posAndNear.x;
  ray.org_y = posAndNear.y;
  ray.org_z = posAndNear.z;
  ray.tnear = posAndNear.w;

  ray.dir_x = dirAndFar.x;
  ray.dir_y = dirAndFar.y;
  ray.dir_z = dirAndFar.z;
  ray.tfar  = dirAndFar.w; // std::numeric_limits<float>::infinity();

  rtcOccluded1(m_scene, &context, &ray);  

  return (ray.tfar < 0.0f);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ISceneObject* CreateEmbreeRT() { return new EmbreeRT; }

ISceneObject* CreateSceneRT(const char* a_impleName) 
{ 
  return CreateEmbreeRT();
}

void DeleteSceneRT(ISceneObject* a_pScene)  { delete a_pScene; }