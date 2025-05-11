#pragma once

static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_max_threads() { return 1; }
static inline int omp_get_thread_num()  { return 0; }
