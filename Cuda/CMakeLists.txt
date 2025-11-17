set(CUDA_PROJECT "Cuda")

find_package(CUDAToolkit REQUIRED)

# Old script from StackOverflow to save in case of need if "native" value won't work
# if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
#     cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto") # optional argument for arch to add
#     message("ARCH_FLAGS = ${ARCH_FLAGS}")
#     string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
#     string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
# else()
#     include(FindCUDA/select_compute_arch)
#     CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
#     string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#     string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
#     string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
#     SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#     set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
# endif()

file(
    GLOB_RECURSE CUDA_CODE_LIST
    LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/*.h*"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda/*.inl"
)

add_executable(${CUDA_PROJECT} ${CUDA_CODE_LIST})

target_compile_features(${CUDA_PROJECT} PRIVATE cxx_std_17)
target_include_directories(${CUDA_PROJECT} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/Utils/include
    ${CMAKE_SOURCE_DIR}/UI/include
    ${CMAKE_SOURCE_DIR}/UI/thirdparty/include
    ${CMAKE_SOURCE_DIR}/UI/thirdparty/include/imgui
    ${CMAKE_SOURCE_DIR}/OpenGL/include
    ${CMAKE_SOURCE_DIR}/OpenGL/thirdparty/include
    ${CMAKE_SOURCE_DIR}/Window/include
    ${CMAKE_SOURCE_DIR}/Window/thirdparty/include
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_features(${CUDA_PROJECT} PRIVATE cxx_std_17 cuda_std_17)
target_link_libraries(${CUDA_PROJECT} UI Utils OpenGL Window CUDA::cudart)
set_target_properties(${CUDA_PROJECT} PROPERTIES CUDA_ARCHITECTURES native)

foreach(CUDA_FILE IN ITEMS ${CUDA_CODE_LIST})
    get_filename_component(CUDA_FILE_PATH "${CUDA_FILE}" PATH)
    file(RELATIVE_PATH CUDA_FILE_RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" "${CUDA_FILE_PATH}")
    string(REPLACE "\\" "/" GROUP_PATH "${CUDA_FILE_RELATIVE_PATH}")
    file(RELATIVE_PATH EXTERNAL_INCLUDE_RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/include")
    string(REPLACE ${EXTERNAL_INCLUDE_RELATIVE_PATH} "" GROUP_PATH "${GROUP_PATH}")
    string(REPLACE "src" "" GROUP_PATH "${GROUP_PATH}")
    string(REPLACE "include/cuda" "" GROUP_PATH "${GROUP_PATH}")
    source_group("${GROUP_PATH}" FILES "${CUDA_FILE}")
endforeach()
