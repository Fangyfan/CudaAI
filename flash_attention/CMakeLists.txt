add_executable(fa_v1 fa_v1.cu)
add_executable(fa_v2 fa_v2.cu)
target_link_libraries(fa_v1 PRIVATE ${CUDA_LIBRARIES} cublas)
target_link_libraries(fa_v2 PRIVATE ${CUDA_LIBRARIES} cublas)
