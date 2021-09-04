#include "glfw_window.h"

#ifdef WIN32
#pragma comment(lib,"glfw3.lib")
#endif
#include <iostream>
#include <stdexcept>
#include <vector>
#include <memory>
#include <cstdint>
#include <sstream>

#include "Camera.h"

#ifdef NDEBUG
constexpr bool g_enableValidationLayers = false;
#else
constexpr bool g_enableValidationLayers = true;
#endif


struct input_sample
{
  bool firstMouse   = true;
  bool captureMouse = false;
  bool capturedMouseJustNow = false;
  bool wireframeMode        = false;

  float lastX, lastY, scrollY;
  float camMoveSpeed     = 1.0f;
  float mouseSensitivity = 0.1f;

} g_inputDesktop;

AppInput g_appInput;

void OnKeyboardPressed_basic(GLFWwindow* window, int key, int scancode, int action, int mode)
{
  switch (key)
  {
    case GLFW_KEY_ESCAPE:
      if (action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, VK_TRUE);
      break;
    case GLFW_KEY_LEFT_SHIFT:
      g_inputDesktop.camMoveSpeed = 10.0f;
      break;

    case GLFW_KEY_LEFT_CONTROL:
      g_inputDesktop.camMoveSpeed = 1.0f;
      break;

    default:
//      if(key >= 0 && key < AppInput::MAXKEYS)
//      {
//        if (action == GLFW_RELEASE)
//          g_appInput.keyReleased[key] = true;
//        else if(action == GLFW_PRESS)
//          g_appInput.keyPressed[key] = true;
//      }
      break;
  }
}

void OnMouseButtonClicked_basic(GLFWwindow* window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    g_inputDesktop.captureMouse = !g_inputDesktop.captureMouse;

  if (g_inputDesktop.captureMouse)
  {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    g_inputDesktop.capturedMouseJustNow = true;
  }
  else
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

}

void OnMouseMove_basic(GLFWwindow* window, double xpos, double ypos)
{
  if (g_inputDesktop.firstMouse)
  {
    g_inputDesktop.lastX      = float(xpos);
    g_inputDesktop.lastY      = float(ypos);
    g_inputDesktop.firstMouse = false;
  }

//  float xoffset = float(xpos) - g_inputDesktop.lastX;
//  float yoffset = g_inputDesktop.lastY - float(ypos);

  g_inputDesktop.lastX = float(xpos);
  g_inputDesktop.lastY = float(ypos);
}

void OnMouseScroll_basic(GLFWwindow* window, double xoffset, double yoffset)
{
  g_inputDesktop.scrollY = float(yoffset);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UpdateCamera(GLFWwindow* a_window, Camera& a_cam, float secondsElapsed)
{
  //move position of camera based on WASD keys, and FR keys for up and down
  if (glfwGetKey(a_window, 'S'))
    a_cam.offsetPosition(secondsElapsed * g_inputDesktop.camMoveSpeed * -1.0f * a_cam.forward());
  else if (glfwGetKey(a_window, 'W'))
    a_cam.offsetPosition(secondsElapsed * g_inputDesktop.camMoveSpeed * a_cam.forward());

  if (glfwGetKey(a_window, 'A'))
    a_cam.offsetPosition(secondsElapsed * g_inputDesktop.camMoveSpeed * -1.0f * a_cam.right());
  else if (glfwGetKey(a_window, 'D'))
    a_cam.offsetPosition(secondsElapsed * g_inputDesktop.camMoveSpeed * a_cam.right());

  if (glfwGetKey(a_window, 'F'))
    a_cam.offsetPosition(secondsElapsed * g_inputDesktop.camMoveSpeed * -1.0f * a_cam.up);
  else if (glfwGetKey(a_window, 'R'))
    a_cam.offsetPosition(secondsElapsed * g_inputDesktop.camMoveSpeed * a_cam.up);

  //rotate camera based on mouse movement
  //
  if (g_inputDesktop.captureMouse)
  {
    if(g_inputDesktop.capturedMouseJustNow)
      glfwSetCursorPos(a_window, 0, 0);

    double mouseX, mouseY;
    glfwGetCursorPos(a_window, &mouseX, &mouseY);
    a_cam.offsetOrientation(g_inputDesktop.mouseSensitivity * float(mouseY), g_inputDesktop.mouseSensitivity * float(mouseX));
    glfwSetCursorPos(a_window, 0, 0); //reset the mouse, so it doesn't go out of the window
    g_inputDesktop.capturedMouseJustNow = false;
  }

  //increase or decrease field of view based on mouse wheel
  //
  const float zoomSensitivity = -0.2f;
  float fieldOfView = a_cam.fov + zoomSensitivity * (float)g_inputDesktop.scrollY;
  if(fieldOfView < 1.0f) fieldOfView   = 1.0f;
  if(fieldOfView > 180.0f) fieldOfView = 180.0f;
  a_cam.fov = fieldOfView;

  g_inputDesktop.scrollY = 0.0f;
}

GLFWwindow* Init(std::shared_ptr<IRender> app, uint32_t a_deviceId, GLFWkeyfun keyboard,
                 GLFWcursorposfun mouseMove,GLFWmousebuttonfun mouseBtn, GLFWscrollfun mouseScroll)
{
  glfwInit();

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  GLFWwindow *window = glfwCreateWindow(app->GetWidth(), app->GetHeight(), "GLFW window", nullptr, nullptr);

  glfwSetKeyCallback        (window, keyboard);
  glfwSetCursorPosCallback  (window, mouseMove);
  glfwSetMouseButtonCallback(window, mouseBtn);
  glfwSetScrollCallback     (window, mouseScroll);
  //glfwSetInputMode          (window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSwapInterval(0);

  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions  = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  app->InitVulkan(glfwExtensions, glfwExtensionCount, a_deviceId);
  VkSurfaceKHR surface;
  VK_CHECK_RESULT(glfwCreateWindowSurface(app->GetVkInstance(), window, nullptr, &surface));

  app->InitPresentation(surface);

  return window;
}


void MainLoop(std::shared_ptr<IRender> &app, GLFWwindow* window)
{
  constexpr int NAverage = 60;
  double avgTime = 0.0;
  int avgCounter = 0;
  int currCam    = 0;

  double lastTime = glfwGetTime();
  while (!glfwWindowShouldClose(window))
  {
    double thisTime = glfwGetTime();
    double diffTime = thisTime - lastTime;
    lastTime        = thisTime;
    
    g_appInput.clearKeys();
    glfwPollEvents();
    
//    if(g_appInput.keyReleased[GLFW_KEY_L])
//      currCam = 1 - currCam;

    UpdateCamera(window, g_appInput.cams[currCam], static_cast<float>(diffTime));
    
    app->ProcessInput(g_appInput);
    app->UpdateCamera(g_appInput.cams, 2);

    app->DrawFrame(static_cast<float>(thisTime));

    // count and print FPS
    //
    avgTime += diffTime;
    avgCounter++;
    if(avgCounter >= NAverage)
    {
      auto title = "test";//app->GetWindowTitle();
      std::stringstream strout;
      strout << "FPS = " << int( 1.0/(avgTime/double(NAverage)) ) << " " << title;

      glfwSetWindowTitle(window, strout.str().c_str());
      avgTime    = 0.0;
      avgCounter = 0;
    }
  }
}
