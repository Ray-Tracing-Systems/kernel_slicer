#ifndef C_IMAGE_PROCESSING_IMAGE_H
#define C_IMAGE_PROCESSING_IMAGE_H


typedef struct image
{
  int w;
  int h;
  int channels;
  unsigned char* data;
} image;

typedef struct image_float
{
  int w;
  int h;
  int channels;
  float* data;
} image_float;

int apply_filter(const char* in_img_path, const char* in_filter_path, const char* out_path);

#endif //C_IMAGE_PROCESSING_IMAGE_H
