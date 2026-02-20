add_executable(naive_transpose naive_transpose.cu)
add_executable(transpose transpose.cu)
target_link_libraries(naive_transpose PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(transpose PRIVATE ${CUDA_LIBRARIES})
