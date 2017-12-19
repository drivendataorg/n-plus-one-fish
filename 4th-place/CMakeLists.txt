cmake_minimum_required(VERSION 3.3)

set(project_name harshml_pipeline)
project(${project_name})

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

find_package(Caffe REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(0_make_sequence_dataset)
add_subdirectory(2_make_cls_global_frame_dataset)
add_subdirectory(4_make_loc_dataset)
add_subdirectory(6_make_cls_roi_dataset)

include_directories(SYSTEM ${Caffe_INCLUDE_DIRS})

add_executable(full_pipeline_inference full_pipeline_inference.cpp)
target_link_libraries(full_pipeline_inference ${OpenCV_LIBS} ${Caffe_LIBRARIES})

add_executable(inference_cls_roi inference_cls_roi.cpp)
target_link_libraries(inference_cls_roi ${OpenCV_LIBS} ${Caffe_LIBRARIES})

add_executable(cls_roi_cleanup cls_roi_cleanup.cpp)
target_link_libraries(cls_roi_cleanup ${OpenCV_LIBS})

add_executable(cls_roi_binarize cls_roi_binarize.cpp)
target_link_libraries(cls_roi_binarize ${OpenCV_LIBS})

add_executable(make_submission make_submission.cpp)
target_link_libraries(make_submission ${OpenCV_LIBS})

add_executable(cls_roi_aggregate cls_roi_aggregate.cpp)
target_link_libraries(cls_roi_aggregate ${OpenCV_LIBS})
