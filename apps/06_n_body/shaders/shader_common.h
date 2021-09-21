#ifndef TEST_SHADER_COMMON_H
#define TEST_SHADER_COMMON_H


const float pointSize = 4.0f;
const float spriteSize = 0.05f;
const float velocityColorScale = 5e-2f;
const float weightColorScale = 0.1f;
//const float velocityColorScale = 1.0f;
const float weightThres = 6.0f;

vec3 colorMap(vec3 velocity, sampler1D colorMapTexture)
{
  float amp = clamp(sqrt(dot(velocity, velocity)) * velocityColorScale, 0.0f, 1.0f);
  return texture(colorMapTexture, amp).rgb;
}

vec3 colorMapWeight(float weight, sampler1D colorMapTexture)
{
  float amp = clamp(weight * weightColorScale, 0.0f, 1.0f);
  return texture(colorMapTexture, amp).rgb;
}

float grayscale(vec3 color)
{
  return color.r * 0.299f + color.g * 0.587f + color.b * 0.114f;
}


#endif //TEST_SHADER_COMMON_H
