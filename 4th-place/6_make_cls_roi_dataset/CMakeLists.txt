include_directories(${OpenCV_INCLUDE_DIR})

set(target_make "make_cls_roi_dataset")

include_directories(SYSTEM ${Caffe_INCLUDE_DIRS})

add_executable(${target_make} make_cls_roi_dataset.cpp)
target_link_libraries(${target_make} ${OpenCV_LIBS} ${Caffe_LIBRARIES})