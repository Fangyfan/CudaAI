add_executable(sgemm sgemm.cu)

target_link_libraries(sgemm PRIVATE ${CUDA_LIBRARIES} cublas)
