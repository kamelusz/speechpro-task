project(speechpro_task)
cmake_minimum_required(VERSION 3.12)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(MSVC)
  add_compile_options(/permissive-)
  add_definitions(-DBOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE -D_WIN32_WINNT=0x0A00)
else(UNIX)
  add_compile_options(-Werror -Wall -O2)
endif(MSVC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)