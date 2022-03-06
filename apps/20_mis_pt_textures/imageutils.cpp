#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <FreeImage.h>

#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>

void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message)
{
  std::cout << "\n** FreeImageError **\n";
  std::cout << message;
  std::cout << "\n** FreeImageError **\n";
}

bool LoadLDRImageFromFile(const wchar_t* a_fileName, int* pW, int* pH, std::vector<uint32_t>& a_data)
{
  FREE_IMAGE_FORMAT fif = FIF_PNG; // image format

  fif = FreeImage_GetFileTypeU(a_fileName, 0);

  if (fif == FIF_UNKNOWN)
    fif = FreeImage_GetFIFFromFilenameU(a_fileName);

  FIBITMAP* dib = nullptr;
  if (FreeImage_FIFSupportsReading(fif))
    dib = FreeImage_LoadU(fif, a_fileName);
  else
  {
    std::cout << "LoadLDRImageFromFile() : FreeImage_FIFSupportsReading/FreeImage_Load failed!" << std::endl;
    return false;
  }

  FIBITMAP* converted = FreeImage_ConvertTo32Bits(dib);
  BYTE* bits          = FreeImage_GetBits(converted);
  auto width          = FreeImage_GetWidth(converted);
  auto height         = FreeImage_GetHeight(converted);
  auto bitsPerPixel   = FreeImage_GetBPP(converted);

  a_data.resize(width*height);
  BYTE* data = (BYTE*)&a_data[0];

  for (unsigned int y = 0; y<height; y++)
  {
    int lineOffset1 = y*width;
    int lineOffset2 = y*width;

    for (unsigned int x = 0; x<width; x++)
    {
      int offset1 = lineOffset1 + x;
      int offset2 = lineOffset2 + x;

      data[4 * offset1 + 0] = bits[4 * offset2 + 2];
      data[4 * offset1 + 1] = bits[4 * offset2 + 1];
      data[4 * offset1 + 2] = bits[4 * offset2 + 0];
      data[4 * offset1 + 3] = bits[4 * offset2 + 3];
    }
  }

  FreeImage_Unload(dib);
  FreeImage_Unload(converted);

  (*pW) = width;
  (*pH) = height;
  return true;
}

bool SaveLDRImageToFile(const wchar_t* a_fileName, int w, int h, uint32_t* data)
{
  FIBITMAP* dib = FreeImage_Allocate(w, h, 32);

  BYTE* bits = FreeImage_GetBits(dib);
  //memcpy(bits, data, w*h*sizeof(int32_t));
  BYTE* data2 = (BYTE*)data;
  for (int i = 0; i<w*h; i++)
  {
    bits[4 * i + 0] = data2[4 * i + 2];
    bits[4 * i + 1] = data2[4 * i + 1];
    bits[4 * i + 2] = data2[4 * i + 0];
    bits[4 * i + 3] = 255; // data2[4 * i + 3]; // 255 to kill alpha channel
  }

	auto imageFileFormat = FIF_PNG;

	std::wstring fileName(a_fileName);
	if (fileName.size() > 4)
	{
		std::wstring resolution = fileName.substr(fileName.size() - 4, 4);

		if (resolution.find(L".bmp") != std::wstring::npos || resolution.find(L".BMP") != std::wstring::npos)
			imageFileFormat = FIF_BMP;
	}

  if (!FreeImage_SaveU(imageFileFormat, dib, a_fileName))
  {
    FreeImage_Unload(dib);
    std::cout << "SaveImageToFile(): FreeImage_Save error on " << a_fileName << std::endl;
    return false;
  }

  FreeImage_Unload(dib);

  return true;
}


bool LoadHDRImageFromFile(const wchar_t* a_fileName, int* pW, int* pH, std::vector<float>& a_data)
{ 
  const wchar_t* filename = a_fileName;
  FREE_IMAGE_FORMAT fif = FIF_UNKNOWN; // image format
  FIBITMAP *dib(NULL), *converted(NULL);
  BYTE* bits(NULL);                    // pointer to the image data
  unsigned int width(0), height(0);    //image width and height

  //check the file signature and deduce its format
  //if still unknown, try to guess the file format from the file extension
  //
  fif = FreeImage_GetFileTypeU(filename, 0);
  if (fif == FIF_UNKNOWN)
  {
    fif = FreeImage_GetFIFFromFilenameU(filename);
  }
  
  if (fif == FIF_UNKNOWN)
  {
    std::cerr << "FreeImage failed to guess file image format: " << filename << std::endl;
    return false;
  }

  //check that the plugin has reading capabilities and load the file
  //
  if (FreeImage_FIFSupportsReading(fif))
  {
    dib = FreeImage_LoadU(fif, filename);
  }
  else
  {
    std::cerr << "FreeImage does not support file image format: " << filename << std::endl;
    return false;
  }

  bool invertY = false; //(fif != FIF_BMP);
  if (!dib)
  {
    std::cerr << "FreeImage failed to load image: " << filename << std::endl;
    return false;
  }

  converted = FreeImage_ConvertToRGBF(dib);
  bits   = FreeImage_GetBits(converted);
  width  = FreeImage_GetWidth(converted);
  height = FreeImage_GetHeight(converted);
  const float* fbits = (const float*)bits;
  a_data.resize(width*height * 4);
  for (unsigned int i = 0; i < width*height; i++)
  {
    a_data[4 * i + 0] = fbits[3 * i + 0];
    a_data[4 * i + 1] = fbits[3 * i + 1];
    a_data[4 * i + 2] = fbits[3 * i + 2];
    a_data[4 * i + 3] = 0.0f;
  }
  if(pW != nullptr) (*pW) = width;
  if(pH != nullptr) (*pH) = height;
  FreeImage_Unload(dib);
  FreeImage_Unload(converted);
  return true;
}
