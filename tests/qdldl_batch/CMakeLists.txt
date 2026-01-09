# QDLDL CUDA Batch Interface Test

if(CUDA_SUPPORT OR CMAKE_CUDA_COMPILER)
    # Test executable
    add_executable(test_qdldl_cuda_batch
        ${CMAKE_CURRENT_SOURCE_DIR}/test_qdldl_cuda_batch.c
    )

    target_include_directories(test_qdldl_cuda_batch PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/lin_sys/direct/qdldl
        ${CMAKE_SOURCE_DIR}/lin_sys/direct/qdldl/qdldl_sources/include
        ${CMAKE_SOURCE_DIR}/lin_sys/direct/kkt_common
        ${CMAKE_SOURCE_DIR}/algebra/csc_tools
        ${CMAKE_BINARY_DIR}/include  # For configured headers
    )

    target_link_libraries(test_qdldl_cuda_batch PRIVATE
        osqpstatic
        stdc++
        m
        cudart
        cublas
        cusparse
    )

    # Enable CUDA device linking for executables that link CUDA static libraries
    set_target_properties(test_qdldl_cuda_batch PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Add to CTest
    add_test(NAME qdldl_cuda_batch_test COMMAND test_qdldl_cuda_batch)

    message(STATUS "QDLDL CUDA batch test enabled")
else()
    message(STATUS "QDLDL CUDA batch test disabled (CUDA not available)")
endif()
