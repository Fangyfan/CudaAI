add_executable(fa flash_attention.cu)
target_link_libraries(fa PRIVATE ${CUDA_LIBRARIES} cublas)