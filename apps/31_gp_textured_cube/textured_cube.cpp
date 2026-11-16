#include "textured_cube.h"
#include "simple_meshes.h"

#include <chrono>
#include <cmath>
#include <string>

static uint32_t ProceduralTexel(int x, int y, int a_size) // checker board with a gradient and dark grid lines, RGBA8
{
  const int   cell    = a_size/8;
  const bool  odd     = ((x/cell) + (y/cell)) % 2 == 1;
  const bool  line    = (x % cell) < 2 || (y % cell) < 2;
  const float fx      = float(x)/float(a_size);
  const float fy      = float(y)/float(a_size);
  float4 color        = odd ? float4(0.95f, 0.85f*fx + 0.15f, 0.2f, 1.0f) : float4(0.15f, 0.35f + 0.5f*fy, 0.9f, 1.0f);
  if(line)
    color = color*0.25f;
  const uint32_t r = uint32_t(color.x*255.0f + 0.5f);
  const uint32_t g = uint32_t(color.y*255.0f + 0.5f);
  const uint32_t b = uint32_t(color.z*255.0f + 0.5f);
  return r | (g << 8) | (b << 16) | 0xFF000000;
}

void TexturedCube::SetViewport(int a_width, int a_height)
{
  const SimpleMeshes::Mesh cube = SimpleMeshes::CreateCube(1.0f, float4(1.0f));
  m_positions = cube.positions;
  m_normals   = cube.normals;
  m_texCoords = cube.texCoords;
  m_indices   = cube.indices;

  const int TEX_SIZE = 256;
  m_texture.resize(TEX_SIZE, TEX_SIZE);
  for(int y = 0; y < TEX_SIZE; y++)
    for(int x = 0; x < TEX_SIZE; x++)
      m_texture[int2(x, y)] = ProceduralTexel(x, y, TEX_SIZE);
  m_sampler.filter   = Sampler::Filter::LINEAR;
  m_sampler.addressU = Sampler::AddressMode::WRAP;
  m_sampler.addressV = Sampler::AddressMode::WRAP;

  const float4x4 view = LiteMath::lookAt(float3(2.6f, 2.2f, 3.4f), float3(0.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f));
  const float4x4 proj = LiteMath::OpenglToVulkanProjectionMatrixFix()*LiteMath::perspectiveMatrix(45.0f, float(a_width)/float(a_height), 0.1f, 100.0f);
  m_viewProj = proj*view;
  m_lightDir = normalize(float4(0.4f, 1.0f, 0.7f, 0.0f));

  m_color.resize(a_width, a_height, LiteImage::Layout::TILED4x4);
  m_depth.resize(a_width, a_height, LiteImage::Layout::TILED4x4);

  m_pipeline.SetColorTarget(m_color);
  m_pipeline.SetDepthTarget(m_depth);
  m_pipeline.SetViewport(0, 0, a_width, a_height);
  m_pipeline.cullMode   = LiteRS::CullMode::BACK;
  m_pipeline.clearColor = float4(0.2f, 0.2f, 0.25f, 1.0f);
}

void TexturedCube::Render(int a_width, int a_height, float a_angle, uint32_t* a_out)
{
  auto start = std::chrono::high_resolution_clock::now();

  LiteRS::Clear(m_pipeline);
  LiteRS::DrawIndexedTriangles(m_pipeline, this, &TexturedCube::kernelVS_Cube, &TexturedCube::kernelPS_Cube, m_indices.data(), uint32_t(m_indices.size()), a_angle);
  kernel2D_ToOutput(a_width, a_height, a_out);

  m_timeRender = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

float4 TexturedCube::kernelVS_Cube(uint32_t a_vertId, float a_angle, TexVaryings* a_out) const
{
  const float4x4 model = LiteMath::rotate4x4Y(a_angle)*LiteMath::rotate4x4X(0.5f*a_angle);
  a_out->normal = model*m_normals[a_vertId];
  a_out->uv     = m_texCoords[a_vertId];
  return m_viewProj*(model*m_positions[a_vertId]);
}

float4 TexturedCube::kernelPS_Cube(const TexVaryings& a_in) const
{
  const float4 texColor = m_texture.sample(m_sampler, a_in.uv);
  const float  diffuse  = std::max(dot(normalize(a_in.normal), m_lightDir), 0.0f);
  const float4 color    = texColor*(0.3f + 0.7f*diffuse);
  return float4(color.x, color.y, color.z, 1.0f);
}

void TexturedCube::kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out)
{
  #pragma omp parallel for
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
      a_out[y*a_width + x] = m_color[int2(x, y)] & 0x00FFFFFF;
  }
}

void TexturedCube::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  a_out[0] = a_out[1] = a_out[2] = a_out[3] = 0.0f;
  if(std::string(a_funcName) == "Render")
    a_out[0] = m_timeRender;
}
