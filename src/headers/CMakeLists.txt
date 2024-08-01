# define library
add_library(
    Improc improc.cpp improc.h
)

# include current directory
target_include_directories(
    Improc
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)



