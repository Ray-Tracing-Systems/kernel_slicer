#include "android_input.h"
#include "render/render_common.h"
#include <cstdint>

AppInput g_appInput;
constexpr float camera_sensitivity = 0.25f;
constexpr float camMoveSpeed       = 1.0f;


int32_t vk_base_android_input_handler(struct android_app* app, AInputEvent* event)
{
  IRender* vk_base_app = reinterpret_cast<IRender*>(app->userData);
  if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
  {
    int32_t eventSource = AInputEvent_getSource(event);
    switch (eventSource) {
      case AINPUT_SOURCE_JOYSTICK:
        //AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_X, 0);
        break;
      case AINPUT_SOURCE_TOUCHSCREEN:
      {
        int32_t action = AMotionEvent_getAction(event);

        switch (action)
        {
          case AMOTION_EVENT_ACTION_UP: {
            // Detect single tap
            int64_t eventTime = AMotionEvent_getEventTime(event);
            int64_t downTime = AMotionEvent_getDownTime(event);
            if (eventTime - downTime <= vk_android::TAP_TIMEOUT)
            {
              float deadZone = (160.f / vk_android::screenDensity) * vk_android::TAP_SLOP * vk_android::TAP_SLOP;
              float x = AMotionEvent_getX(event, 0) - g_appInput.touch_pos_x;
              float y = AMotionEvent_getY(event, 0) - g_appInput.touch_pos_y;
              if ((x * x + y * y) < deadZone)
              {
                g_appInput.keyPressed[0] = true;
                g_appInput.cams[0].offsetPosition(g_appInput.cams[0].forward() * camMoveSpeed);
              }
            };

            return 1;
          }
          case AMOTION_EVENT_ACTION_DOWN: {
            // Detect double tap
            int64_t eventTime = AMotionEvent_getEventTime(event);
            if (eventTime - g_appInput.lastTapTime <= vk_android::DOUBLE_TAP_TIMEOUT)
            {
              float deadZone = (160.f / vk_android::screenDensity) * vk_android::DOUBLE_TAP_SLOP * vk_android::DOUBLE_TAP_SLOP;
              float x = AMotionEvent_getX(event, 0) - g_appInput.touch_pos_x;
              float y = AMotionEvent_getY(event, 0) - g_appInput.touch_pos_y;
              if ((x * x + y * y) < deadZone)
              {
                g_appInput.keyPressed[1] = true;
                g_appInput.touchDown = false;
              }
            }
            else
            {
              g_appInput.touchDown = true;
            }
            g_appInput.touch_pos_x = AMotionEvent_getX(event, 0);
            g_appInput.touch_pos_y = AMotionEvent_getY(event, 0);
            break;
          }
          case AMOTION_EVENT_ACTION_MOVE: {

            int32_t eventX = AMotionEvent_getX(event, 0);
            int32_t eventY = AMotionEvent_getY(event, 0);

            float deltaX = (float)(g_appInput.touch_pos_x - eventX) * camera_sensitivity;
            float deltaY = (float)(g_appInput.touch_pos_y - eventY) * camera_sensitivity;

            g_appInput.cams[0].offsetOrientation(deltaY, deltaX);

            g_appInput.touch_pos_x = eventX;
            g_appInput.touch_pos_y = eventY;
            break;
          }
          default:
            return 1;
        }
      }
      default:
        return 1;
    }
  }

  return 0;
}