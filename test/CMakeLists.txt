project(speechpro-test)
cmake_minimum_required(VERSION 3.12)

add_executable(${PROJECT_NAME} 
  common.hpp
  random_matrix.cpp
  precalculated_matrix.cpp
  main.cpp)

add_dependencies(${PROJECT_NAME} speechpro)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CONAN_INCLUDE_DIRS_GTEST}
    ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} 
  PRIVATE
    speechpro
    CONAN_PKG::gtest)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
file(COPY "./data" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})