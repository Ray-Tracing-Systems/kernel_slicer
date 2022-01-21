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
    std::cout << "Time (CPU) : " <<  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds" << std::endl;
  }

  SaveBMP("test.bmp", output.data(), WIDTH, HEIGHT);
  
  /*
  int grid_resolution[3] = {grid_res.x, grid_res.y, grid_res.z};
  auto cam_pos = marcher.GetCamPos();
  auto cam_mat = marcher.GetInvProjViewMat();

  float camera[3] = { cam_pos.x, cam_pos.y, cam_pos.z};
  float ray_matrix[4][4];
  ray_matrix[0][0] = cam_mat.get_row(0).x;
  ray_matrix[0][1] = cam_mat.get_row(0).y;
  ray_matrix[0][2] = cam_mat.get_row(0).z;
  ray_matrix[0][3] = cam_mat.get_row(0).w;

  ray_matrix[1][0] = cam_mat.get_row(1).x;
  ray_matrix[1][1] = cam_mat.get_row(1).y;
  ray_matrix[1][2] = cam_mat.get_row(1).z;
  ray_matrix[1][3] = cam_mat.get_row(1).w;

  ray_matrix[2][0] = cam_mat.get_row(2).x;
  ray_matrix[2][1] = cam_mat.get_row(2).y;
  ray_matrix[2][2] = cam_mat.get_row(2).z;
  ray_matrix[2][3] = cam_mat.get_row(2).w;

  ray_matrix[3][0] = cam_mat.get_row(3).x;
  ray_matrix[3][1] = cam_mat.get_row(3).y;
  ray_matrix[3][2] = cam_mat.get_row(3).z;
  ray_matrix[3][3] = cam_mat.get_row(3).w;

  {
    auto start = std::chrono::steady_clock::now();
    ispc::Execute_ISPC(density.data(), grid_resolution, camera, ray_matrix, WIDTH, HEIGHT, output.data());
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time (ISPC): " <<  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds" << std::endl;
  }

  SaveBMP("test_ispc.bmp", output.data(), WIDTH, HEIGHT);

  {
    auto start = std::chrono::steady_clock::now();
    ispc::Execute_ISPC_multithreaded(density.data(), grid_resolution, camera, ray_matrix, WIDTH, HEIGHT, output.data());
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time (ISPC multithreaded): " <<  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds" << std::endl;
  }

  SaveBMP("test_ispc_multithreaded.bmp", output.data(), WIDTH, HEIGHT);
  */

  return 0;
}
