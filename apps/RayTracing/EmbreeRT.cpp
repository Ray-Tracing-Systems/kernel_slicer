#include <iostream>
#include <vector>
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

  void BeginScene() override; 
  void EndScene  () override; 
  
  uint32_t AddInstance(uint32_t a_geomId, const float a_matrixData[16], bool a_rowMajor = false) override;
  void     UpdateInstance(uint32_t a_instanceId, uint32_t a_geomId, const float* a_matrixData, bool a_rowMajor = false) override;

  CRT_Hit  RayQuery(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override;

protected:
  RTCDevice m_device = nullptr;
  RTCScene  m_scene  = nullptr;
  
  uint32_t  m_currGeomTop = 0;
  //std::vector<RTCGeometry> m_geoms;
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
  m_device = rtcNewDevice("isa=avx512");
  m_scene  = rtcNewScene(m_device);
  rtcSetDeviceErrorFunction(m_device, error_handler, nullptr);
  //m_geoms.reserve(1000);
}

EmbreeRT::~EmbreeRT()
{
  rtcReleaseScene(m_scene);
  rtcReleaseDevice(m_device);
}

void EmbreeRT::ClearGeom()
{
  rtcReleaseScene(m_scene);
  m_scene = rtcNewScene(m_device);
  m_currGeomTop = 0;
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

  // In rtcAttachGeometry(...), the scene takes ownership of the geom by increasing its reference count. This means that we don't have
  // to hold on to the geom handle, and may release it. The geom object will be released automatically when the scene is destroyed.//
  // rtcAttachGeometry() returns a geometry ID. We could use this to identify intersected objects later on.
  //
  uint32_t geomId = rtcAttachGeometry(m_scene, geom);
  rtcReleaseGeometry(geom);

  assert(geomId == m_currGeomTop);
  
  m_currGeomTop++;
  return m_currGeomTop-1;
}

void EmbreeRT::UpdateGeom_Triangles4f(uint32_t a_geomId, const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber)
{
  std::cout << "EmbreeRT::UpdateGeom_Triangles4f is not implemented yet!" << std::endl;  
}

void EmbreeRT::BeginScene()
{

} 

void EmbreeRT::EndScene()
{
  rtcCommitScene(m_scene);
}  

uint32_t EmbreeRT::AddInstance(uint32_t a_geomId, const float a_matrixData[16], bool a_rowMajor)
{
  return 0;
}

void     EmbreeRT::UpdateInstance(uint32_t a_instanceId, uint32_t a_geomId, const float* a_matrixData, bool a_rowMajor)
{

}

CRT_Hit  EmbreeRT::RayQuery(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar)
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
  rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  // There are multiple variants of rtcIntersect. This one intersects a single ray with the scene.
  // 
  rtcIntersect1(m_scene, &context, &rayhit);
  
  CRT_Hit result;
  if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
  {
    result.t      = rayhit.ray.tfar;
    result.geomId = rayhit.hit.geomID;
    result.instId = rayhit.hit.instID[0];
    result.primId = rayhit.hit.primID;
    result.coords[0] = rayhit.hit.u;
    result.coords[1] = rayhit.hit.v;
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ISceneObject* CreateSceneRT(const char* a_impleName) { return new EmbreeRT; }
void          DeleteSceneRT(ISceneObject* a_pScene)  { delete a_pScene; }
