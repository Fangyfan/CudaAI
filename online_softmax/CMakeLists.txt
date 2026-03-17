add_executable(online_softmax online_softmax.cu)
target_link_libraries(online_softmax PRIVATE ${CUDA_LIBRARIES})
