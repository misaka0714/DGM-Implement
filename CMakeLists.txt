cmake_minimum_required(VERSION 3.16)

project(
  dgm_algorithm
  VERSION 0.1.0
  DESCRIPTION "Cross-model graph ANN index migration"
  LANGUAGES CXX)

option(DGM_BUILD_EXAMPLES "Build the minimal DGM example" ON)
option(DGM_BUILD_TESTS "Build DGM tests" ON)

find_package(Threads REQUIRED)

add_library(
  dgm
  src/embedding_space.cpp
  src/graph.cpp
  src/local_migration.cpp
  src/neighbor_selection.cpp
  src/packed_signatures.cpp
  src/search_migration.cpp)
add_library(dgm::dgm ALIAS dgm)

target_compile_features(dgm PUBLIC cxx_std_17)
target_include_directories(
  dgm
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dgm PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dgm PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(DGM_BUILD_EXAMPLES)
  add_executable(dgm_minimal examples/minimal.cpp)
  target_link_libraries(dgm_minimal PRIVATE dgm::dgm)
endif()

if(DGM_BUILD_TESTS)
  enable_testing()
  add_executable(dgm_tests tests/dgm_test.cpp)
  target_link_libraries(dgm_tests PRIVATE dgm::dgm)
  add_test(NAME dgm_tests COMMAND dgm_tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(
  TARGETS dgm
  EXPORT dgmTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
  EXPORT dgmTargets
  NAMESPACE dgm::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dgm)

configure_package_config_file(
  cmake/dgmConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/dgmConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dgm)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/dgmConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/dgmConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/dgmConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dgm)
install(FILES LICENSE NOTICE README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
