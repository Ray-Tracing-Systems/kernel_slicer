#pragma once
#include <cstddef>

void* aligned_alloc (size_t __alignment, size_t __size);
void* malloc (size_t __size);
void  free(void* data);

namespace std 
{
  void* aligned_alloc (size_t __alignment, size_t __size);
  void* malloc (size_t __size);
  void  free(void* data);
};

static constexpr int EXIT_FAILURE =	1;	/* Failing exit status.  */
static constexpr int EXIT_SUCCESS =	0;	/* Successful exit status.  */