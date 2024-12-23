cmake_minimum_required(VERSION 3.13)
project(ortcxx VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------------
# Setup dependencies.
# -----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntimeConfig.cmake)
find_package(onnxruntime)

#* ONNXANROID
# file(GLOB LIB_ONNXRUNTIME_FILES 
# ${PROJECT_SOURCE_DIR}/libs/onnxruntime-android/jni/${ANDROID_ABI}/*.so
# )
# message("LIB_ONNXRUNTIME_FILES: ${LIB_ONNXRUNTIME_FILES}")


# -----------------------------------------------------------------------------
# Build library
# -----------------------------------------------------------------------------
set(LIB_NAME ortcxx)
set(LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/modules)
set(LIB_PRIV_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/headers)
file(GLOB LIB_SRC_FILES 
  ${LIB_SRC_DIR}/**/*.cpp
)
add_library(${LIB_NAME} SHARED ${LIB_HEADERS} ${LIB_PRIV_HEADERS} ${LIB_SRC_FILES})
target_link_libraries(${LIB_NAME} onnxruntime)

## ANDROID
# target_link_libraries(${LIB_NAME} ${LIB_ONNXRUNTIME_FILES})
# target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/libs/onnxruntime-android/headers)

# 
target_include_directories(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${LIB_HEADERS}> $<INSTALL_INTERFACE:include>)
target_include_directories(${LIB_NAME} PRIVATE $<BUILD_INTERFACE:${LIB_HEADERS}/${LIB_NAME}> $<INSTALL_INTERFACE:include/${LIB_NAME}>)
target_include_directories(${LIB_NAME} PRIVATE $<BUILD_INTERFACE:${LIB_PRIV_HEADERS}> $<INSTALL_INTERFACE:include/${LIB_NAME}>)
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# -----------------------------------------------------------------------------
# Buil executable and tests.
# -----------------------------------------------------------------------------
add_executable(hello tests/hello.cpp)
add_executable(pipeline tests/inference.cpp)
target_link_libraries(hello ortcxx)
target_link_libraries(pipeline ortcxx)

#-----------------------------------------------------------------------------
# Install header and library.
#-----------------------------------------------------------------------------
## ANDROID
install(TARGETS ${LIB_NAME} hello pipeline
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY ${LIB_HEADERS}/ DESTINATION include)

install(DIRECTORY ${LIB_PRIV_HEADERS}/ DESTINATION include/${LIB_NAME})