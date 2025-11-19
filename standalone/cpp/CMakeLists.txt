cmake_minimum_required(VERSION 3.0)
project(AgeEstimation)

find_package(OpenCV REQUIRED)

add_executable(AgeDemo age.cpp)
target_link_libraries(AgeDemo ${OpenCV_LIBS})

add_executable(PoseDemo pose.cpp)
target_link_libraries(PoseDemo ${OpenCV_LIBS})