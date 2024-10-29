cmake_minimum_required(VERSION 3.0.2)
project(m2t2_ros)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  tf
  rospy
  geometry_msgs
  sensor_msgs
  std_msgs
)

add_message_files(
  FILES
  GraspCandidate.msg
  GraspCandidates.msg
)

add_service_files(
  FILES
  GraspPrediction.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(CATKIN_DEPENDS
  message_runtime
  tf
  rospy
  geometry_msgs
  std_msgs
  sensor_msgs
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
