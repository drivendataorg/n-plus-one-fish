include_directories(${OpenCV_INCLUDE_DIR})

set(target_make "make_loc_dataset")

add_executable(${target_make} make_loc_dataset.cpp)
target_link_libraries(${target_make} ${OpenCV_LIBS})