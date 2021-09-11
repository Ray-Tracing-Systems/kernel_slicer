#ifndef TEST_SHADER_COMMON_H
#define TEST_SHADER_COMMON_H


const float pointSize = 4.0f;
const float spriteSize = 0.025f;
const float velocityColorScale = 5e-3f;
//const float velocityColorScale = 1.0f;

vec3 colorMap(vec3 velocity, sampler1D colorMapTexture)
{
  float amp = clamp(sqrt(dot(velocity, velocity)) * velocityColorScale, 0.0f, 1.0f);
  return texture(colorMapTexture, amp).rgb;
}

#endif //TEST_SHADER_COMMON_H
