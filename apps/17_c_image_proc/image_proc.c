#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include "image_proc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

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

char g_pathPrefix[PATH_BUF] = {};

int tex_clamp(int idx, int upper_thres)
{
  if(idx < 0)
    return 0;

  if(idx >= upper_thres)
    return upper_thres - 1;

  return idx;
}

typedef int (*WrapMode)(int, int);
typedef pix3 (*TextureSampleUC3)(image*, float, float, WrapMode);
typedef pix3 (*TexelFetchUC3)(image*, int, int, WrapMode);
typedef float (*TexelFetchF1)(image_float*, int, int, WrapMode);
typedef void (*ImageStoreUC3)(image*, int, int, pix3 color);


float texel_fetch_float1(image_float *img, int u, int v, WrapMode mode)
{
  int x = mode(u, img->w);
  int y = mode(v, img->h);

  int idx = y * img->w + x;

  return img->data[idx];
}

pix3 texel_fetch_uc(image *img, int u, int v, WrapMode mode)
{
  int x = mode(u, img->w);
  int y = mode(v, img->h);

  int idx = y * img->w + x;
  pix3 res = {{img->data[idx * 3 + 0],
               img->data[idx * 3 + 1],
               img->data[idx * 3 + 2]} };

  return res;
}

void image_store_uc(image* img, int x, int y, pix3 color)
{
  img->data[(y * img->w + x) * 3 + 0] = color.color[0];
  img->data[(y * img->w + x) * 3 + 1] = color.color[1];
  img->data[(y * img->w + x) * 3 + 2] = color.color[2];
}

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
      {
        for (int l = -half; l <= half; ++l)
        {
          float filter_val = texel_fetch_float1(filter, k + half, l + half, tex_clamp);

          pix3 x = texel_fetch_uc(in, j + l, i + k, tex_clamp);

          madd_pixels(&acc, filter_val, &x);
        }
      }
      pix3 res_color = {{acc.color[0], acc.color[1], acc.color[2]}};
      image_store_uc(out, j, i, res_color);
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


// ping pong filter chain
void do_computations(image* in, int filter_num, image_float filters[filter_num], image* out, bool save_intermediate)
{
  char path_buf[PATH_BUF * 2] = {};
  image temp = create_image(in->w, in->h, in->channels);

  image* current_in  = in;
  image* current_out = &temp;
  for(int i = 0; i < filter_num - 1; ++i)
  {
    kernel2D_filter(current_in, &filters[i], current_out);

    if(save_intermediate)
    {
      snprintf(path_buf, PATH_BUF * 2, "%s/step_%d.png", g_pathPrefix, i);
      save_image(path_buf, current_out);
    }

    image *p = current_in;
    current_in = current_out;
    current_out = p;
  }

  kernel2D_filter(current_in, &filters[filter_num - 1], out);

  free(temp.data);
}

void set_path_prefix(const char* out_path)
{
  char* pLastSlash = strrchr(out_path, '/');
  if(pLastSlash != NULL)
  {
    strncpy(g_pathPrefix, out_path, strlen(out_path) - strlen(pLastSlash));
  }
  else
  {
    g_pathPrefix[0] = '.';
  }
}

int apply_filters(const char* in_img_path, const char* out_path, const char** filter_paths, int filters_num,
                  bool save_intermediate)
{
  image in     = {};
  image out    = {};
  image_float filters[filters_num];

  load_image(in_img_path, &in);
  out = create_image(in.w, in.h, in.channels);

  for(int i = 0; i < filters_num; ++i)
  {
    if (load_filter(filter_paths[i], &filters[i]))
    {
      fprintf(stderr, "Failed loading filter from file: %s\n", filter_paths[i]);
      return 1;
    }
  }

  if(save_intermediate)
    set_path_prefix(out_path);

  do_computations(&in, filters_num, filters, &out, save_intermediate);

  save_image(out_path, &out);
  fprintf(stdout, "Written output image: %s\n", out_path);

  stbi_image_free(in.data);
  free(out.data);

  for(int i = 0; i < filters_num; ++i)
  {
    free(filters[i].data);
  }

  return 0;
}