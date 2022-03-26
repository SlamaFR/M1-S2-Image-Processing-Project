cmake_minimum_required(VERSION 2.8)

# find opencv
find_package(OpenCV REQUIRED)
# compilation flags
set(CMAKE_CXX_FLAGS "-Wall -g -O2 -std=c++11")

# or the automatic version
file(GLOB_RECURSE SRC_FILES src/*.cpp)

# for each sample file, make an exe

add_executable(project ${SRC_FILES})
target_link_libraries(project ${OpenCV_LIBS})

