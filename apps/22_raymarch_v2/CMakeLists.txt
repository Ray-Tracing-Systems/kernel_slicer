cmake_minimum_required(VERSION 3.16)
project(raymarch_sample)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ISPC_COMPILER /home/frol/ispc/bin)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddISPCExample.cmake)

if(WIN32)
    add_compile_definitions(WIN32)
    if(MSVC)
        add_compile_options(/EHsc)
    endif()
endif()


set (EXAMPLES_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

set (ISPC_SRC_NAME "raymarch") # => raymarch.ispc
set (TARGET_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/raymarch.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/Bitmap.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
set (ISPC_IA_TARGETS "sse2-i32x4,sse4-i32x8,avx1-i32x8,avx2-i32x8,avx512knl-i32x16,avx512skx-i32x16" CACHE STRING "ISPC IA targets")
set (ISPC_ARM_TARGETS "neon" CACHE STRING "ISPC ARM targets")
set (DATA_FILES )
#${CMAKE_CURRENT_SOURCE_DIR}/camera.dat
#${CMAKE_CURRENT_SOURCE_DIR}/density_highres.vol
#${CMAKE_CURRENT_SOURCE_DIR}/density_lowres.vol

include_directories("../vkutils"
                    "../volk" 
                    "../LiteMath" 
                    "../loader_utils"
                    "../geo_utils"
                    "../utils")

#add_ispc_example(NAME "raymarch"
#        ISPC_IA_TARGETS ${ISPC_IA_TARGETS}
#        ISPC_ARM_TARGETS ${ISPC_ARM_TARGETS}
#        ISPC_SRC_NAME ${ISPC_SRC_NAME}
#        TARGET_SOURCES ${TARGET_SOURCES}
#        USE_COMMON_SETTINGS
#        DATA_FILES ${DATA_FILES}
#        )

add_executable(raymarch_sample main.cpp
                               raymarch.cpp
                               Bitmap.cpp)
