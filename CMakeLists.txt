cmake_minimum_required(VERSION 3.5)
project(human_pose_estimation)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  human_pose_estimation/pose_estimation.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
