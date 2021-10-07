#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "image_proc.h"
#include "stdio.h"
#include "stdlib.h"

typedef struct pix3
{
  unsigned char color[3];
} pix3;

typedef struct ipix3
{
  int color[3];
} ipix3;

typedef struct fpix3
{
  float color[3];
} fpix3;

// only LDR for now
int load_image(const char* a_path, image* out)
{
  assert(!out->data);
//  stbi_info(a_path, &out->w, &out->h, &out->channels);
  out->channels = 3;
  int actual_channels;
  unsigned char* pixels = stbi_load(a_path, &out->w, &out->h, &actual_channels, out->channels);

  if(out->w <= 0 || out->h <= 0 || !pixels)
  {
    fprintf(stderr, "Error loading image from %s\n", a_path);
    return 1;
  }

  out->data = pixels;

  return 0;
}

image create_image(int w, int h, int channels)
{
  image res = {.w = w, .h = h, .channels = channels, .data = NULL};

  res.data = calloc(w * h * channels, sizeof(unsigned char ));
  if(res.data == NULL)
  {
    fprintf(stderr, "create_image: memory allocation error\n");
  }

  return res;
}

image_float create_image_float(int w, int h, int channels)
{
  image_float res = {.w = w, .h = h, .channels = channels, .data = NULL};

  res.data = calloc(w * h * channels, sizeof(float));
  if(res.data == NULL)
  {
    fprintf(stderr, "create_image_float: memory allocation error\n");
  }

  return res;
}

int save_image(const char* path, image* img)
{
  return stbi_write_png(path, img->w, img->h, img->channels, img->data, img->w * img->channels);
}

void madd_pixels(fpix3* sum, float mult, pix3* add)
{
  sum->color[0] += mult * add->color[0];
  sum->color[1] += mult * add->color[1];
  sum->color[2] += mult * add->color[2];
}

int get_boundary_index(int idx, int upper_thres)
{
  if(idx < 0)
    return 0;

  if(idx >= upper_thres)
    return upper_thres - 1;

  return idx;
}

void kernel2D_filter(image* in, image_float* filter, image* out)
{
  assert(in->data && out->data);
  assert(in->w == out->w && in->h == out->h);
  assert(filter->w < in->w && filter->h < in->h);

  const int half = filter->w / 2;

  for(int i = 0; i < in->h; ++i)
  {
    for(int j = 0; j < in->w; ++j)
    {
      fpix3 acc = { {0.0f, 0.0f, 0.0f} };
      for(int k = -half; k <= half; ++k)
        for(int l = -half; l <= half; ++l)
        {                                                                        ///<! Issue (#4):
          float filter_val = filter->data[(k + half) * filter->w + (l + half)];  ///<! explicit indexing will not work for GPU images in this way
          int offset_x = get_boundary_index(i + k, in->h);                       ///<! think we don't have to do this for GPU images also
          int offset_y = get_boundary_index(j + l, in->w);                       ///<! think we don't have to do this for GPU images also

          int idx = offset_x * in->w + offset_y;
          pix3 x = { {in->data[idx * 3 + 0],             ///<! explicit indexing will not work for GPU images in this way
                      in->data[idx * 3 + 1],             ///<! explicit indexing will not work for GPU images in this way
                      in->data[idx * 3 + 2]} };          ///<! explicit indexing will not work for GPU images in this way

          madd_pixels(&acc, filter_val, &x);
        }                                                ///<! ///<! Issue (#5):
      out->data[(i * in->w + j) * 3 + 0] = acc.color[0]; ///<! explicit indexing will not work for GPU images in this way
      out->data[(i * in->w + j) * 3 + 1] = acc.color[1]; ///<! explicit indexing will not work for GPU images in this way
      out->data[(i * in->w + j) * 3 + 2] = acc.color[2]; ///<! explicit indexing will not work for GPU images in this way
    }
  }
}

int load_filter(const char* path, image_float* filter)
{
  FILE *fp = fopen(path, "r");
  if(fp == NULL)
  {
    fprintf(stderr, "error opening filter file");
    return 1;
  }
  char * line = NULL;
  size_t len = 0;

  ssize_t read = getline(&line, &len, fp);
  if(read == -1)
    return 1;

  char *end;
  int filter_size = strtol(line, &end, 10);
  *filter = create_image_float(filter_size, filter_size, 1);

  for(int i = 0; i < filter_size; ++i)
  {
    read = getline(&line, &len, fp);
    if(read == -1)
      return 1;

    char* begin = line;
    for(int j = 0; j < filter_size; ++j)
    {
      filter->data[i * filter_size + j] = strtof(begin, &end);
      begin = end;
    }
  }

  free(line);
  fclose(fp);

  return 0;
}

int apply_filter(const char* in_img_path, const char* in_filter_path, const char* out_path) ///<! Q: is this a 'control' function analogue?
{
  image in     = {};
  image out    = {};
  image_float filter = {};
  load_image(in_img_path, &in);                ///<! Issue (#1) allocate memory, load from file
  out = create_image(in.w, in.h, in.channels); ///<! Issue (#2) allocate memory. How do we translate these to Vulkan?
  if (load_filter(in_filter_path, &filter))
  {
    fprintf(stderr, "Failed loading filter from file: %s\n", in_filter_path);
    return 1;
  }

  kernel2D_filter(&in, &filter, &out);         ///<! Q: do we suppose automatic data transfer before first and after last kernels?

  save_image(out_path, &out);                  ///<! Issue (#3) copy data back .... 
  fprintf(stdout, "Written output image: %s\n", out_path);

  stbi_image_free(in.data); ///<! Issue (#4) WTF?
  free(out.data);           ///<! how should we distinguish 
  free(filter.data);        ///<! these 2 cases?

  return 0;
}
