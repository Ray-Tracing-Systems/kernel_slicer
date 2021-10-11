#ifndef C_IMAGE_PROCESSING_IMAGE_H
#define C_IMAGE_PROCESSING_IMAGE_H


#include <stdbool.h>

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

enum
{
  PATH_BUF = 128
};

int apply_filter(const char* in_img_path, const char* in_filter_path, const char* out_path);
int apply_filters(const char* in_img_path, const char* out_path, const char** filter_paths, int filters_num,
                  bool save_intermediate);

#endif //C_IMAGE_PROCESSING_IMAGE_H
