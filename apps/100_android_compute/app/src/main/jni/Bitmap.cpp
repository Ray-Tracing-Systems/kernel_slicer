#include <vector>
#include <fstream>
#include <cstring>
#include <android_native_app_glue.h>
#include "vk_utils.h"

namespace vk_android
{
  extern AAssetManager *g_pMgr;
}

#define BMP_HEADER_LEN 54

struct Pixel { unsigned char r, g, b; };

void WriteBMP(const char* fname, Pixel* a_pixelData, int width, int height)
{
  int paddedsize = (width*height) * sizeof(Pixel);

  unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
  unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};

  bmpfileheader[ 2] = (unsigned char)(paddedsize    );
  bmpfileheader[ 3] = (unsigned char)(paddedsize>> 8);
  bmpfileheader[ 4] = (unsigned char)(paddedsize>>16);
  bmpfileheader[ 5] = (unsigned char)(paddedsize>>24);

  bmpinfoheader[ 4] = (unsigned char)(width    );
  bmpinfoheader[ 5] = (unsigned char)(width>> 8);
  bmpinfoheader[ 6] = (unsigned char)(width>>16);
  bmpinfoheader[ 7] = (unsigned char)(width>>24);
  bmpinfoheader[ 8] = (unsigned char)(height    );
  bmpinfoheader[ 9] = (unsigned char)(height>> 8);
  bmpinfoheader[10] = (unsigned char)(height>>16);
  bmpinfoheader[11] = (unsigned char)(height>>24);

  std::ofstream out(fname, std::ios::out | std::ios::binary);
  out.write((const char*)bmpfileheader, 14);
  out.write((const char*)bmpinfoheader, 40);
  out.write((const char*)a_pixelData, paddedsize);
  out.flush();
  out.close();
}

void SaveBMPAndroid(const char* fname, const unsigned int* pixels, int w, int h)
{
  std::vector<Pixel> pixels2(w*h);

  for (size_t i = 0; i < pixels2.size(); i++)
  {
    Pixel px;
    px.r       = (pixels[i] & 0x00FF0000) >> 16;
    px.g       = (pixels[i] & 0x0000FF00) >> 8;
    px.b       = (pixels[i] & 0x000000FF);
    pixels2[i] = px;
  }

  WriteBMP(fname, &pixels2[0], w, h);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<unsigned int> LoadBMPAndroid(const char* filename, int* pW, int* pH)
{
  assert(vk_android::g_pMgr);
  auto file = AAssetManager_open(vk_android::g_pMgr, filename, AASSET_MODE_STREAMING);

  if(file == NULL)
  {
    (*pW) = 0;
    (*pH) = 0;
    return std::vector<unsigned int>();
  }

  unsigned char info[BMP_HEADER_LEN];
  auto read_bytes = AAsset_read(file, info, BMP_HEADER_LEN * sizeof(unsigned char));
  if(!read_bytes)
    RUN_TIME_ERROR("[LoadBMP]: AAsset_read error (header)");

  int width  = *(int*)&info[18];
  int height = *(int*)&info[22];

  int row_padded = (width*3 + 3) & (~3);
  auto data      = new unsigned char[row_padded];

  std::vector<unsigned int> res(width*height);
  for(int i = 0; i < height; i++)
  {
    read_bytes = AAsset_read(file, data, row_padded * sizeof(unsigned char));
    if(!read_bytes)
      RUN_TIME_ERROR("[LoadBMP]: AAsset_read error (row_padded)");

    for(int j = 0; j < width; j++)
    {
      res[i * width + j] = (uint32_t(data[j * 3 + 0]) << 16) | (uint32_t(data[j * 3 + 1]) << 8) |
                           (uint32_t(data[j * 3 + 2]) << 0);
    }
  }

  AAsset_close(file);
  delete [] data;

  (*pW) = width;
  (*pH) = height;
  return res;
}