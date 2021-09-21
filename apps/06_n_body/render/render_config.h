#ifndef TEST_RENDER_CONFIG_H
#define TEST_RENDER_CONFIG_H

enum class RENDER_MODE
{
  POINTS,
  SPRITES
};

static constexpr RENDER_MODE DISPLAY_MODE  = RENDER_MODE::SPRITES;
static const char* SPRITE_TEXTURE_PATH     = "textures/sphere.png";


#endif //TEST_RENDER_CONFIG_H
