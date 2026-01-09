# Add CUDA-PCG

# Find CUDA samples helper headers
set(CUDA_SAMPLES_COMMON_DIR "" CACHE PATH "Path to CUDA samples Common directory")
if(NOT CUDA_SAMPLES_COMMON_DIR)
    # Try common locations
    if(EXISTS "/home/chiang/opt/cuda-samples/Common")
        set(CUDA_SAMPLES_COMMON_DIR "/home/chiang/opt/cuda-samples/Common")
    elseif(EXISTS "/usr/local/cuda/samples/common/inc")
        set(CUDA_SAMPLES_COMMON_DIR "/usr/local/cuda/samples/common/inc")
    elseif(EXISTS "$ENV{HOME}/cuda-samples/Common")
        set(CUDA_SAMPLES_COMMON_DIR "$ENV{HOME}/cuda-samples/Common")
    endif()
endif()

set(cuda_pcg_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_SAMPLES_COMMON_DIR}
)

set(cuda_pcg_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_pcg_interface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda_pcg_interface.c
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_pcg_constants.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_pcg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_pcg.cu
)

# Create object library for linear system solver interface
add_library(linsys_cuda_pcg OBJECT ${cuda_pcg_sources})

target_include_directories(linsys_cuda_pcg PRIVATE
                           ${cuda_pcg_includes}
                           ${osqp_api_includes}
                           ${algebra_impl_includes}
                           ${csc_tools_includes})

# Pass upwards
set(cuda_pcg_includes ${cuda_pcg_includes} PARENT_SCOPE)
