add_executable(topk topk.cu)
target_link_libraries(topk PRIVATE ${CUDA_LIBRARIES})
