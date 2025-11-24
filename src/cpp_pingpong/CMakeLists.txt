cmake_minimum_required(VERSION 3.8)
project(cpp_pingpong)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pingpong_msgs REQUIRED)

add_executable(controller src/controller.cpp)
add_executable(ballistics src/ballistics.cpp)

ament_target_dependencies(controller ballistics rclcpp pingpong_msgs)
ament_package()

install(TARGETS
  controller
  DESTINATION lib/${PROJECT_NAME}
)