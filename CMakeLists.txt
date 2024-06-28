cmake_minimum_required(VERSION 3.13)
project(cinnamon VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------------
# Setup dependencies.
# -----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntimeConfig.cmake)
find_package(onnxruntime)

# -----------------------------------------------------------------------------
# Build library
# -----------------------------------------------------------------------------
set(LIB_NAME cinnamon)
set(LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/modules)
set(LIB_PRIV_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/headers)
file(GLOB LIB_SRC_FILES 
  ${LIB_SRC_DIR}/**/*.cpp
)
add_library(${LIB_NAME} SHARED ${LIB_HEADERS} ${LIB_PRIV_HEADERS} ${LIB_SRC_FILES})
target_link_libraries(${LIB_NAME} onnxruntime OpenCV)
target_include_directories(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${LIB_HEADERS}> $<INSTALL_INTERFACE:include>)
target_include_directories(${LIB_NAME} PRIVATE $<BUILD_INTERFACE:${LIB_HEADERS}/${LIB_NAME}> $<INSTALL_INTERFACE:include/${LIB_NAME}>)
target_include_directories(${LIB_NAME} PRIVATE $<BUILD_INTERFACE:${LIB_PRIV_HEADERS}> $<INSTALL_INTERFACE:include/${LIB_NAME}>)
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# -----------------------------------------------------------------------------
# Buil executable and tests.
# -----------------------------------------------------------------------------
add_executable(hello tests/hello.cpp)
target_link_libraries(hello cinnamon)