#include <iostream>
#include <chrono>
#include "Bitmap.h"
#include "raymarch.h"
//#include "raymarch_ispc.h"


std::vector<float> voxel_sphere(LiteMath::int3 resolution, float radius)
{
  std::vector<float> result(resolution.x * resolution.y * resolution.z, 0.0f);
  LiteMath::int3 center = resolution / 2 - 1;

  for(int z = 0; z < resolution.z; ++z)
  {
    for(int y = 0; y < resolution.y; ++y)
    {
      for(int x = 0; x < resolution.x; ++x)
      {
        if((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) + (z - center.z) * (z - center.z) <= radius * radius)
          result[z * resolution.x * resolution.y + y * resolution.x + x] = 1.0f;
      }
    }
  }

  return result;
}


int main()
{
  constexpr uint32_t WIDTH = 512;
  constexpr uint32_t HEIGHT = 512;
  RayMarcher marcher(WIDTH, HEIGHT);

  auto grid_res = LiteMath::int3(100, 100, 100);

//  std::vector<float> density (grid_res.x * grid_res.y * grid_res. z, 1.0f);
  auto density = voxel_sphere(grid_res, 25.0f);
  marcher.Init(density, grid_res);

  std::vector<uint32_t> output(HEIGHT * WIDTH, 0);

  {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < HEIGHT; ++i)
    {
      for (int j = 0; j < WIDTH; ++j)
      {
        marcher.Execute(i, j, output.data());
      }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time (CPU) : " <<  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " microseconds" << std::endl;
  }

  SaveBMP("test.bmp", output.data(), WIDTH, HEIGHT);

  return 0;
}
