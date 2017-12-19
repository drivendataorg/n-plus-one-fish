include_directories(${OpenCV_INCLUDE_DIR})

set(target_parse "parse_sequence_dataset")
set(target_make "make_sequence_dataset")

add_executable(${target_parse} parse_sequence_dataset.cpp)
target_link_libraries(${target_parse} ${OpenCV_LIBS})

add_executable(${target_make} make_sequence_dataset.cpp)
target_link_libraries(${target_make} ${OpenCV_LIBS})