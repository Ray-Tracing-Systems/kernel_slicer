cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 17)

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()


if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -fopenmp")

add_executable(test main.cpp test_class.cpp Bitmap.cpp)

set_target_properties(test PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

