project(speechpro)
cmake_minimum_required(VERSION 3.12)

add_library(${PROJECT_NAME} STATIC
  integral_processing.hpp
  integral_processing.cpp
  processing_context.hpp
  processing_context.cpp)

target_link_libraries(${PROJECT_NAME} 
  PUBLIC
    CONAN_PKG::boost
    CONAN_PKG::opencv)
  
target_include_directories(${PROJECT_NAME}
  INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  PUBLIC 
    ${CONAN_INCLUDE_DIRS_BOOST}
    ${CONAN_INCLUDE_DIRS_OPENCV})