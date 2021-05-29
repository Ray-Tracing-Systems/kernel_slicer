#version 330

layout(location = 0) in vec2 vertex;

void main(void)
{
  gl_Position = vec4(vertex,0.0,1.0);
  gl_Position.y = -gl_Position.y;	// Vulkan coordinate system is different t OpenGL    
}
