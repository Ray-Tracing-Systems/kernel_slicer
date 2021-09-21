#ifndef CBVH_STF_ANDROID_INPUT_H
#define CBVH_STF_ANDROID_INPUT_H

#include <android_native_app_glue.h>

int32_t vk_base_android_input_handler(struct android_app* app, AInputEvent* event);

namespace vk_android
{
  const int32_t DOUBLE_TAP_TIMEOUT = 300 * 1000000;
  const int32_t TAP_TIMEOUT = 180 * 1000000;
  const int32_t DOUBLE_TAP_SLOP = 100;
  const int32_t TAP_SLOP = 8;
}

#endif //CBVH_STF_ANDROID_INPUT_H
