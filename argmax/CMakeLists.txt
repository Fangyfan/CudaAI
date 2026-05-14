add_executable(argmax argmax.cu)
target_link_libraries(argmax PRIVATE ${CUDA_LIBRARIES})
