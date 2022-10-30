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
  std::cout << "\n***\n";
  std::cout << message;
  std::cout << "\n***\n";
}

bool LoadLDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<int32_t>& a_data)
{
  FREE_IMAGE_FORMAT fif = FIF_PNG; // image format

  fif = FreeImage_GetFileType(a_fileName, 0);

  if (fif == FIF_UNKNOWN)
    fif = FreeImage_GetFIFFromFilename(a_fileName);

  FIBITMAP* dib = nullptr;
  if (FreeImage_FIFSupportsReading(fif))
    dib = FreeImage_Load(fif, a_fileName);
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
}

bool SaveLDRImageToFile(const char* a_fileName, int w, int h, int32_t* data)
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

	std::string fileName(a_fileName);
	if (fileName.size() > 4)
	{
		std::string resolution = fileName.substr(fileName.size() - 4, 4);

		if (resolution.find(".bmp") != std::string::npos || resolution.find(".BMP") != std::wstring::npos)
			imageFileFormat = FIF_BMP;
	}

  if (!FreeImage_Save(imageFileFormat, dib, a_fileName))
  {
    FreeImage_Unload(dib);
    std::cout << "SaveImageToFile(): FreeImage_Save error on " << a_fileName << std::endl;
    return false;
  }

  FreeImage_Unload(dib);

  return true;
}


bool LoadHDRImageFromFile(const char* a_fileName, 
                          int* pW, int* pH, std::vector<float>& a_data)
{
  
    const char* filename = a_fileName;

    FREE_IMAGE_FORMAT fif = FIF_UNKNOWN; // image format
    FIBITMAP *dib(NULL), *converted(NULL);
    BYTE* bits(NULL);                    // pointer to the image data
    unsigned int width(0), height(0);    //image width and height

    //check the file signature and deduce its format
    //if still unknown, try to guess the file format from the file extension
    //
    fif = FreeImage_GetFileType(filename, 0);
  
    
    if (fif == FIF_UNKNOWN)
    {
      fif = FreeImage_GetFIFFromFilename(filename);
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
      dib = FreeImage_Load(fif, filename);
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


float MSE_RGB_LDR(const std::vector<int32_t>& image1, const std::vector<int32_t>& image2)
{
  if(image1.size() != image2.size())
    return 0.0f;

  double accum = 0.0;

  for(int i=0;i<image1.size();i++)
  {
    const int pxData1 = image1[i];
    const int pxData2 = image2[i];
    const int r1 = (pxData1 & 0x00FF0000) >> 16;
    const int g1 = (pxData1 & 0x0000FF00) >> 8;
    const int b1 = (pxData1 & 0x000000FF);

    const int r2 = (pxData2 & 0x00FF0000) >> 16;
    const int g2 = (pxData2 & 0x0000FF00) >> 8;
    const int b2 = (pxData2 & 0x000000FF);

    accum += double( (r1-r2)*(r1-r2) + (b1-b2)*(b1-b2) + (g1-g2)*(g1-g2) );
  }

  return float(accum/double(image1.size()));
}


float MSE_RGB_HDR(const std::vector<float>& image1, const std::vector<float>& image2)
{
  if(image1.size() != image2.size())
    return 0.0f;

  double accum = 0.0;

  for(int i=0;i<image1.size();i+=4)
  {
    const float r1 = image1[i+0];
    const float g1 = image1[i+1];
    const float b1 = image1[i+2];
    
    const float r2 = image2[i+0];
    const float g2 = image2[i+1];
    const float b2 = image2[i+2];

    accum += double( (r1-r2)*(r1-r2) + (b1-b2)*(b1-b2) + (g1-g2)*(g1-g2) );
  }

  return float(4.0*accum/double(image1.size())); // we mult by 4 due to image2.size() == w*h*4, but we actually want w*h
}



