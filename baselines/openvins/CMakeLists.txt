cmake_minimum_required(VERSION 3.16)
project(openvins_euroc_runner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenVINS v2.7 (commit 93adc241) with openvins_export_only.patch applied,
# built without ROS. See README.md in this folder.
set(OPENVINS_ROOT "" CACHE PATH "OpenVINS source checkout")
set(OPENVINS_BUILD "" CACHE PATH "Folder containing libov_msckf_lib.so")
if(NOT OPENVINS_ROOT OR NOT OPENVINS_BUILD)
  message(FATAL_ERROR "Set -DOPENVINS_ROOT=... and -DOPENVINS_BUILD=...")
endif()

find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs video features2d calib3d highgui flann)
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Ceres 2.1 REQUIRED)

add_library(openvins_core SHARED IMPORTED)
set_target_properties(openvins_core PROPERTIES
  IMPORTED_LOCATION "${OPENVINS_BUILD}/libov_msckf_lib.so"
  INTERFACE_INCLUDE_DIRECTORIES
    "${OPENVINS_ROOT}/ov_msckf/src;${OPENVINS_ROOT}/ov_core/src;${OPENVINS_ROOT}/ov_init/src"
)

add_executable(
  euroc_file_runner
  euroc_file_runner.cpp
  openvins_state_exporter.cpp
  openvins_feature_exporter.cpp
)
target_include_directories(euroc_file_runner PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(euroc_file_runner PRIVATE openvins_core Eigen3::Eigen ${OpenCV_LIBS} ${Boost_LIBRARIES} Ceres::ceres)
target_compile_options(euroc_file_runner PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(euroc_file_runner PROPERTIES BUILD_RPATH "${OPENVINS_BUILD}")
