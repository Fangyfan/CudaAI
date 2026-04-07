add_executable(sgemm sgemm.cu)
add_executable(roofline roofline_bench.cu)

target_link_libraries(sgemm PRIVATE ${CUDA_LIBRARIES} cublas)
target_link_libraries(roofline PRIVATE ${CUDA_LIBRARIES} cublas)