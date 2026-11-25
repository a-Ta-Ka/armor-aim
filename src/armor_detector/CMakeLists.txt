cmake_minimum_required(VERSION 3.0.2)
project(armor_detector)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  image_transport
  nodelet
  tf2
  tf2_ros
  geometry_msgs
  message_generation
  dynamic_reconfigure
  visualization_msgs
)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_message_files(
  FILES
  Detection.msg
  DetectionFrame.msg
  TrackTarget.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)
generate_dynamic_reconfigure_options(
  cfg/ArmorDetector.cfg
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES armor_detector
 CATKIN_DEPENDS cv_bridge roscpp rospy sensor_msgs std_msgs image_transport nodelet tf2 tf2_ros geometry_msgs message_runtime dynamic_reconfigure visualization_msgs

)

include_directories(
include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
    src/detector.cpp
    src/tracker.cpp
    src/nn_detector.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    openvino::runtime
)

add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

