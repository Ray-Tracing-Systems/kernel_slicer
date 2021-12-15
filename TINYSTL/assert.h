#pragma once

void _check_assert(bool x);

#define assert(expr) _check_assert(expr)
