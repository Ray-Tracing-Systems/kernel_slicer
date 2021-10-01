#include <stdio.h>
#include <stdlib.h>
#include "image_proc.h"


int main(int argc, char** argv)
{
  if(argc != 4)
  {
    printf("Wrong arguments. Usage:\n");
    printf("c_image_proc input_image_path filter_path save_path\n");
    printf("filter file is a text file, first line contains size, "
           "following lines contain rows of a square matrix of that size, example:\n");
    printf("3\n-1 -1 -1\n -1 8 -1\n-1 -1 -1\n");
    exit(1);
  }

  apply_filter(argv[1], argv[2], argv[3]);

  return 0;
}
