typedef float<3> float3;
typedef float<4> float4;
typedef int<3> int3;

struct Ray {
  float3 origin, dir;
};

static void InitEyeRay(float x, float y, uniform int w, uniform int h,
                       uniform float3 cam_pos, const uniform float ray_matrix[4][4],
                       Ray& ray)
{
  ray.origin.x = 0.0f;
  ray.origin.y = 0.0f;
  ray.origin.z = 10.0f;

  float4 pos;
  pos.x = 2.0f * (x + 0.5f) / w - 1.0f;
  pos.y = -2.0f * (y + 0.5f) / h + 1.0f;
  pos.z = 0.0f;
  pos.w = 1.0f;

  pos.x = pos.x * ray_matrix[0][0] + pos.y * ray_matrix[0][1] + pos.z * ray_matrix[0][2] + pos.w * ray_matrix[0][3];
  pos.y = pos.x * ray_matrix[1][0] + pos.y * ray_matrix[1][1] + pos.z * ray_matrix[1][2] + pos.w * ray_matrix[1][3];
  pos.z = pos.x * ray_matrix[2][0] + pos.y * ray_matrix[2][1] + pos.z * ray_matrix[2][2] + pos.w * ray_matrix[2][3];
  pos.w = pos.x * ray_matrix[3][0] + pos.y * ray_matrix[3][1] + pos.z * ray_matrix[3][2] + pos.w * ray_matrix[3][3];

  pos = pos / pos.w; //pos.x = pos.x / pos.w; pos.y = pos.y / pos.w; pos.z = pos.z / pos.w;
  pos.y *= (-1.0f);
  float len = sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
  float lenInv = 1.0f / len;
  pos = pos * lenInv;

  ray.dir.x = pos.x;
  ray.dir.y = pos.y;
  ray.dir.z = pos.z;
}

static bool RayBoxIntersection(Ray ray, float3 boxMin, float3 boxMax, float &tmin, float &tmax)
{
  float3 raydir = ray.dir;
  raydir.x = 1.0f / raydir.x;
  raydir.y = 1.0f / raydir.y;
  raydir.z = 1.0f / raydir.z;

  float lo = raydir.x*(boxMin.x - ray.origin.x);
  float hi = raydir.x*(boxMax.x - ray.origin.x);

  tmin = min(lo, hi);
  tmax = max(lo, hi);

  float lo1 = raydir.y*(boxMin.y - ray.origin.y);
  float hi1 = raydir.y*(boxMax.y - ray.origin.y);

  tmin = max(tmin, min(lo1, hi1));
  tmax = min(tmax, max(lo1, hi1));

  float lo2 = raydir.z*(boxMin.z - ray.origin.z);
  float hi2 = raydir.z*(boxMax.z - ray.origin.z);

  tmin = max(tmin, min(lo2, hi2));
  tmax = min(tmax, max(lo2, hi2));

  return (tmin <= tmax) && (tmax > 0.f);
}

static inline float Lerp(float t, float a, float b) {
  return (1.f - t) * a + t * b;
}

static inline bool Inside(float3 p, float3 pMin, float3 pMax)
{
  return (p.x >= pMin.x && p.x <= pMax.x && p.y >= pMin.y && p.y <= pMax.y && p.z >= pMin.z && p.z <= pMax.z);
}

static inline float GetDensity(int x, int y, int z, uniform int3 resolution, uniform float density[])
{
  x = clamp(x, 0, resolution.x - 1);
  y = clamp(y, 0, resolution.y - 1);
  z = clamp(z, 0, resolution.z - 1);

#pragma ignore warning(perf)
  return density[z * resolution.x * resolution.y + y * resolution.x + x];
}


static inline float3 Normalize(float3 p, float3 pMin, float3 pMax)
{
  return (p - pMin) / (pMax - pMin);
}


static float SampleDensity(float3 pos, float3 pMin, float3 pMax, uniform float density[], uniform int3 resolution)
{
  if (!Inside(pos, pMin, pMax))
    return 0;

  float3 vox = Normalize(pos, pMin, pMax);
  vox.x = vox.x * resolution.x - .5f;
  vox.y = vox.y * resolution.y - .5f;
  vox.z = vox.z * resolution.z - .5f;
  int vx = (int)(vox.x), vy = (int)(vox.y), vz = (int)(vox.z);
  float dx = vox.x - vx, dy = vox.y - vy, dz = vox.z - vz;

  float d00 = Lerp(dx, GetDensity(vx,   vy,   vz, resolution, density),
                       GetDensity(vx+1, vy,   vz, resolution, density));
  float d10 = Lerp(dx, GetDensity(vx,   vy+1, vz, resolution, density),
                       GetDensity(vx+1, vy+1, vz, resolution, density));
  float d01 = Lerp(dx, GetDensity(vx,   vy,   vz+1, resolution, density),
                       GetDensity(vx+1, vy,   vz+1, resolution, density));
  float d11 = Lerp(dx, GetDensity(vx,   vy+1, vz+1, resolution, density),
                       GetDensity(vx+1, vy+1, vz+1, resolution, density));
  float d0 = Lerp(dy, d00, d10);
  float d1 = Lerp(dy, d01, d11);
  return Lerp(dz, d0, d1);
}

static float RayMarch(uniform float density[], uniform int3 resolution, Ray ray)
{
  float tmin, tmax;
  uniform float3 SCENE_BOX_MIN = {-3, -3, -3};
  uniform float3 SCENE_BOX_MAX = {+3, +3, +3};

  cif (!RayBoxIntersection(ray, SCENE_BOX_MIN, SCENE_BOX_MAX, tmin, tmax))
    return 0.;

  tmin = max(tmin, 0.f);

  uniform float sigma_a = 10;
  uniform float sigma_s = 10;
  uniform float dt = 0.025;

  float alpha = 0.f;

  float acc = 0.0f;
  float t = tmin;
  int n = 0;
  cwhile (t < tmax)
  {
    if(exp(-alpha) < .005f)
      break;

    float3 pos = ray.origin + t * ray.dir;
    float d = SampleDensity(pos, SCENE_BOX_MIN, SCENE_BOX_MAX, density, resolution);

    acc += d;
    alpha += dt * (sigma_a + sigma_s) * d;
    t += dt;
    n += 1;
  }

  return acc / n;
}

static uint32 color_convert(float3 color)
{
#pragma ignore warning(perf)
  uint32 R = 255 * color.x;
#pragma ignore warning(perf)
  uint32 G = 255 * color.y;
#pragma ignore warning(perf)
  uint32 B = 255 * color.z;

  uint32 res_color = 0xFF000000 | R | (G << 8) | (B << 16);

  return res_color;
}

static void march_tile(uniform int x0, uniform int y0, uniform int x1, uniform int y1,
                       uniform float density[], uniform int3 resolution,
                       uniform float3 cam_pos, const uniform float ray_matrix[4][4],
                       uniform int width, uniform int height, uniform unsigned int image[])
{
  // 4x4 = 16 pixel tiles
  // implicitly assumes that (x1-x0) and (y1-y0) are evenly divisible by 4.

  for (uniform int y = y0; y < y1; y += 4) {
    for (uniform int x = x0; x < x1; x += 4) {
      foreach (o = 0 ... 16) {
        // These two arrays encode the mapping from [0,15] to
        // offsets within the 4x4 pixel block so that we render
        // each pixel inside the block
        const uniform int xoffsets[16] = { 0, 1, 0, 1, 2, 3, 2, 3,
                                           0, 1, 0, 1, 2, 3, 2, 3 };
        const uniform int yoffsets[16] = { 0, 0, 1, 1, 0, 0, 1, 1,
                                           2, 2, 3, 3, 2, 2, 3, 3 };

        int xo = x + xoffsets[o];
        int yo = y + yoffsets[o];

        Ray ray;
        InitEyeRay(xo, yo, width, height, cam_pos, ray_matrix, ray);

        int offset = yo * width + xo;
#pragma ignore warning(perf)
        image[offset] = color_convert(RayMarch(density, resolution, ray));
      }
    }
  }
}

task void march_task(uniform int dx, uniform int dy, uniform float density[], uniform int3 resolution,
                     uniform float3 cam_pos, const uniform float ray_matrix[4][4],
                     uniform int width, uniform int height, uniform unsigned int out_color[])
{
  uniform int xbuckets = (width + (dx-1)) / dx;
  uniform int ybuckets = (height + (dy-1)) / dy;

  uniform int x0 = (taskIndex % xbuckets) * dx;
  uniform int y0 = (taskIndex / xbuckets) * dy;
  uniform int x1 = x0 + dx;
  uniform int y1 = y0 + dy;
  x1 = min(x1, width);
  y1 = min(y1, height);

  march_tile(x0, y0, x1, y1, density, resolution, cam_pos, ray_matrix, width, height, out_color);
}

export void Execute_ISPC(uniform float density_field[], uniform int grid_resolution[3],
                         uniform float cam_pos[3], const uniform float ray_matrix[4][4],
                         uniform int width, uniform int height, uniform unsigned int out_color[])
{

  uniform int3 resolution;
  resolution.x = grid_resolution[0];
  resolution.y = grid_resolution[1];
  resolution.z = grid_resolution[2];

  uniform float3 camera_pos;
  camera_pos.x = cam_pos[0];
  camera_pos.y = cam_pos[1];
  camera_pos.z = cam_pos[2];

  march_tile(0, 0, width, height, density_field, resolution, camera_pos, ray_matrix, width, height, out_color);
}

export void Execute_ISPC_multithreaded(uniform float density_field[], uniform int grid_resolution[3],
                                       uniform float cam_pos[3], const uniform float ray_matrix[4][4],
                                       uniform int width, uniform int height, uniform unsigned int out_color[])
{
  uniform int dx = 8, dy = 8;
  uniform int nTasks = ((width + (dx - 1)) / dx) * ((height + (dy - 1)) / dy);

  uniform int3 resolution;
  resolution.x = grid_resolution[0];
  resolution.y = grid_resolution[1];
  resolution.z = grid_resolution[2];

  uniform float3 camera_pos;
  camera_pos.x = cam_pos[0];
  camera_pos.y = cam_pos[1];
  camera_pos.z = cam_pos[2];

  launch[nTasks] march_task(dx, dy, density_field, resolution, camera_pos, ray_matrix, width, height, out_color);
}
