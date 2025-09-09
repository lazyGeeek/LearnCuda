set(LEARN_CUDA_PROJECT "LearnCuda")

file(
    GLOB_RECURSE LEARN_CUDA_CODE_LIST
    LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/learn_cuda/*.h*"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/learn_cuda/*.inl"
)

add_executable(${LEARN_CUDA_PROJECT} ${LEARN_CUDA_CODE_LIST})

target_compile_features(${LEARN_CUDA_PROJECT} PRIVATE cxx_std_17)
target_include_directories(${LEARN_CUDA_PROJECT} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/Cuda/include
    ${CMAKE_SOURCE_DIR}/Utils/include
    ${CMAKE_SOURCE_DIR}/Window/include
    ${CMAKE_SOURCE_DIR}/Window/thirdparty/include
)
target_link_libraries(${LEARN_CUDA_PROJECT} Cuda Utils Window)

foreach(LEARN_CUDA_FILE IN ITEMS ${LEARN_CUDA_CODE_LIST})
    get_filename_component(LEARN_CUDA_FILE_PATH "${LEARN_CUDA_FILE}" PATH)
    file(RELATIVE_PATH LEARN_CUDA_FILE_RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" "${LEARN_CUDA_FILE_PATH}")
    string(REPLACE "\\" "/" GROUP_PATH "${LEARN_CUDA_FILE_RELATIVE_PATH}")
    file(RELATIVE_PATH EXTERNAL_INCLUDE_RELATIVE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/" "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/include")
    string(REPLACE ${EXTERNAL_INCLUDE_RELATIVE_PATH} "" GROUP_PATH "${GROUP_PATH}")
    string(REPLACE "src" "" GROUP_PATH "${GROUP_PATH}")
    string(REPLACE "include/learn_cuda" "" GROUP_PATH "${GROUP_PATH}")
    source_group("${GROUP_PATH}" FILES "${LEARN_CUDA_FILE}")
endforeach()
