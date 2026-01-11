# Add qdldl
add_subdirectory(qdldl_sources)

if(NOT DEFINED EMBEDDED)
set(
    amd_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd_internal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/SuiteSparse_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_aat.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_control.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_defaults.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_info.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_order.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_post_tree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_postorder.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_preprocess.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_valid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/SuiteSparse_config.c
)
endif()

set(qdldl_interface_sources
    ${amd_sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.c
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_symbolic.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_symbolic.c
)


set(qdldl_interface_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_sources/include
)

# Create object library for linear system solver interface
add_library(linsys_qdldl OBJECT ${qdldl_interface_sources})

target_include_directories(linsys_qdldl PRIVATE
                           ${qdldl_interface_includes}
                           ${csc_tools_includes}
                           ${osqp_api_includes}
                           ${kkt_common_includes})

# Optional: CUDA batched GPU solver
if(CUDA_SUPPORT OR CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    set(qdldl_batch_gpu_sources
        ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_batch_gpu.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_batch_gpu.cuh
        ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_batch_gpu.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_cuda_batch_interface.h
        ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_cuda_batch_interface.c
    )

    add_library(linsys_qdldl_batch_gpu OBJECT ${qdldl_batch_gpu_sources})

    target_include_directories(linsys_qdldl_batch_gpu PRIVATE
                               ${qdldl_interface_includes}
                               ${csc_tools_includes}
                               ${osqp_api_includes}
                               ${kkt_common_includes})

    set_target_properties(linsys_qdldl_batch_gpu PROPERTIES
                          CUDA_SEPARABLE_COMPILATION ON
                          POSITION_INDEPENDENT_CODE ON)

    message(STATUS "QDLDL batched GPU solver enabled")

    # ADMM batched GPU kernels (separate from linear solver)
    set(admm_batch_gpu_sources
        ${CMAKE_SOURCE_DIR}/include/admm_batch_gpu.h
        ${CMAKE_SOURCE_DIR}/src/admm_batch_gpu.cu
    )

    add_library(admm_batch_gpu OBJECT ${admm_batch_gpu_sources})

    target_include_directories(admm_batch_gpu PRIVATE
                               ${qdldl_interface_includes}
                               ${csc_tools_includes}
                               ${osqp_api_includes}
                               ${kkt_common_includes})

    set_target_properties(admm_batch_gpu PROPERTIES
                          CUDA_SEPARABLE_COMPILATION ON
                          POSITION_INDEPENDENT_CODE ON)

    message(STATUS "ADMM batched GPU kernels enabled")
endif()

