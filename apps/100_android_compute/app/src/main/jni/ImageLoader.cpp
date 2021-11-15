#include <vector>
#include <fstream>
#include <cstring>
#include <android_native_app_glue.h>
#include "vk_utils.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#include "lodepng.h"

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

bool LoadEXRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data)
{
  assert(vk_android::g_pMgr);
  auto file = AAssetManager_open(vk_android::g_pMgr, a_fileName, AASSET_MODE_BUFFER);

  if(file == NULL)
  {
    (*pW) = 0;
    (*pH) = 0;
    a_data = {};
    return false;
  }

  size_t fileLength = AAsset_getLength(file);

  std::vector<unsigned char> bytes(fileLength, 0.0f);
  auto read_bytes = AAsset_read(file, bytes.data(), fileLength);
  if(!read_bytes)
    RUN_TIME_ERROR("[vk_utils::readSPVFile]: AAsset_read error");
  AAsset_close(file);

  EXRHeader header;
  EXRVersion version;
  EXRImage image;
  const char *error;
  auto memory = static_cast<unsigned char *>(bytes.data());

//  float** data;
//  auto status = LoadEXRFromMemory(data, pW, pH, memory, fileLength, error);
//  if (status != TINYEXR_SUCCESS) {
//    std::stringstream  ss;
//    ss << "Failed to load EXR from " << a_fileName;
//    FreeEXRErrorMessage(*error);
//    RUN_TIME_ERROR(ss.str().c_str());
//  }
//


  InitEXRHeader(&header);
  InitEXRImage(&image);

  auto status = ParseEXRVersionFromMemory(&version, memory, read_bytes);
  if (status != TINYEXR_SUCCESS) {
    std::stringstream  ss;
    ss << "Failed to parse EXR version for file " << a_fileName;
    FreeEXRErrorMessage(error);
    RUN_TIME_ERROR(ss.str().c_str());
  }

  status = ParseEXRHeaderFromMemory(&header, &version, memory, read_bytes, &error);
  if( status != TINYEXR_SUCCESS )
  {
    std::stringstream  ss;
    ss << "Failed to parse OpenEXR header for file " << a_fileName << ": "  << error;
    FreeEXRErrorMessage(error);
    RUN_TIME_ERROR(ss.str().c_str());
  }

  for (int i = 0; i < header.num_channels; i++) {
    if (header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
      header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }
  }

  status = LoadEXRImageFromMemory(&image, &header, memory, read_bytes, &error );
  if( status != TINYEXR_SUCCESS )
  {
    std::stringstream  ss;
    ss << "Failed to parse OpenEXR file " << a_fileName << ": "  << error;
    FreeEXRErrorMessage(error);
    RUN_TIME_ERROR(ss.str().c_str());
  }

  // RGBA
  int idxR = -1;
  int idxG = -1;
  int idxB = -1;
  int idxA = -1;
  for (int c = 0; c < header.num_channels; c++) {
    if (strcmp(header.channels[c].name, "R") == 0) {
      idxR = c;
    }
    else if (strcmp(header.channels[c].name, "G") == 0) {
      idxG = c;
    }
    else if (strcmp(header.channels[c].name, "B") == 0) {
      idxB = c;
    }
    else if (strcmp(header.channels[c].name, "A") == 0) {
      idxA = c;
    }
  }

  *pW = image.width;
  *pH = image.height;
  a_data.resize(image.width * image.height * image.num_channels);
  for (size_t i = 0; i < image.width * image.height; i++) {
    auto ptr = reinterpret_cast<float **>(image.images);
    a_data[4 * i + 0] = ptr[idxR][i];
    a_data[4 * i + 1] = ptr[idxG][i];
    a_data[4 * i + 2] = ptr[idxB][i];
    a_data[4 * i + 3] = idxA != -1 ? ptr[idxA][i] : 1.0f;
  }

  FreeEXRImage(&image);

  return true;
}

bool LoadLDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<int32_t>& a_data)
{
  AAsset* file = AAssetManager_open(vk_android::g_pMgr, a_fileName, AASSET_MODE_BUFFER);
  size_t fileLength = AAsset_getLength(file);

  std::vector<unsigned char> png(fileLength, 0);

  auto read_bytes = AAsset_read(file, png.data(), fileLength);
  if(!read_bytes)
    RUN_TIME_ERROR("[loadPNG]: AAsset_read error");
  AAsset_close(file);

  std::vector<unsigned char> image; //the raw pixels
  unsigned width, height;

  unsigned error = lodepng::decode(image, width, height, png);
  //if there's an error, display it
  if(error)
  {
    std::stringstream ss;
    ss << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    RUN_TIME_ERROR(ss.str().c_str());
  }

  a_data.resize(width*height);
  for(size_t i = 0; i < a_data.size(); ++i)
  {
    auto red   = image[i * 4 + 0];
    auto green = image[i * 4 + 1];
    auto blue  = image[i * 4 + 2];
    auto alpha = image[i * 4 + 3];
    a_data[i] = red | (green << 8) | (blue << 16) | alpha;
  }

  *pW = static_cast<int>(width);
  *pH = static_cast<int>(height);

  return true;
}