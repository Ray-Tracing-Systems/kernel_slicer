#ifndef CBVH_STF_GLFW_WINDOW_H
#define CBVH_STF_GLFW_WINDOW_H

#include "render/render_common.h"

#include "GLFW/glfw3.h"
#include <memory>
#include <unordered_map>


void OnKeyboardPressed_basic(GLFWwindow* window, int key, int scancode, int action, int mode);
void OnMouseButtonClicked_basic(GLFWwindow* window, int button, int action, int mods);
void OnMouseMove_basic(GLFWwindow* window, double xpos, double ypos);
void OnMouseScroll_basic(GLFWwindow* window, double xoffset, double yoffset);

GLFWwindow * Init(std::shared_ptr<IRender> app, uint32_t a_deviceId = 0,
                  GLFWkeyfun keyboard = OnKeyboardPressed_basic,
                  GLFWcursorposfun mouseMove = OnMouseMove_basic,
                  GLFWmousebuttonfun mouseBtn = OnMouseButtonClicked_basic,
                  GLFWscrollfun mouseScroll = OnMouseScroll_basic);
void MainLoop(std::shared_ptr<IRender> &app, GLFWwindow* window);

std::unordered_map<std::string, std::string> ReadCommandLineParams(int argc, const char** argv);

#endif //CBVH_STF_GLFW_WINDOW_H
