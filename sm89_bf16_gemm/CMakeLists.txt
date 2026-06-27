add_executable(test test.cu bf16_gemm.cu)

set_property(TARGET test PROPERTY CUDA_ARCHITECTURES 89)

target_link_libraries(test PRIVATE ${CUDA_LIBRARIES} cublas)