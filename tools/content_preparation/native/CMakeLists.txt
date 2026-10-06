cmake_minimum_required(VERSION 3.16)
project(content_preparation_draco_bridge LANGUAGES CXX)

set(DRACO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/draco" CACHE PATH "Pinned Draco source")
get_filename_component(DRACO_SOURCE_DIR "${DRACO_SOURCE_DIR}" ABSOLUTE)
if(NOT EXISTS "${DRACO_SOURCE_DIR}/src/draco/core/draco_version.h")
  message(FATAL_ERROR "DRACO_SOURCE_DIR must point to the initialized pinned Draco submodule")
endif()
set(DRACO_TESTS OFF CACHE BOOL "" FORCE)
set(DRACO_BUILD_EXECUTABLES OFF CACHE BOOL "" FORCE)
set(DRACO_INSTALL OFF CACHE BOOL "" FORCE)
set(DRACO_TRANSCODER_SUPPORTED OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
add_subdirectory("${DRACO_SOURCE_DIR}" draco-build EXCLUDE_FROM_ALL)

find_package(Git REQUIRED)
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${DRACO_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE CP_DRACO_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE CP_GIT_RESULT)
if(NOT CP_GIT_RESULT EQUAL 0)
  message(FATAL_ERROR "Cannot record the pinned Draco source commit")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${DRACO_SOURCE_DIR}" diff --quiet HEAD
  RESULT_VARIABLE CP_DIRTY_RESULT)
if(NOT CP_DIRTY_RESULT EQUAL 0)
  message(FATAL_ERROR "Draco tracked source differs from its commit; restore/pin the intended source before building")
endif()
add_executable(draco_byte_codec draco_byte_codec.cpp)
target_compile_features(draco_byte_codec PRIVATE cxx_std_17)
target_include_directories(draco_byte_codec PRIVATE
  "${DRACO_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(draco_byte_codec PRIVATE draco::draco)
target_compile_definitions(draco_byte_codec PRIVATE
  CP_DRACO_COMMIT="${CP_DRACO_COMMIT}"
  CP_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
  CP_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
