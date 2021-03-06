# Copyright (c) 2017 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

################################################################################
set(phylanx_SOURCES "" CACHE INTERNAL "Sources for libphylanx." FORCE)
if(MSVC)
  set(phylanx_HEADERS "" CACHE INTERNAL "Headers for libphylanx." FORCE)
endif()

################################################################################
# gather sources/headers for main library

# libphylanx sources
add_phylanx_library_sources(phylanx
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_phylanx_library_sources(phylanx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/ast/*.cpp"
  APPEND)
add_phylanx_library_sources(phylanx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/execution_tree/*.cpp"
  APPEND)
add_phylanx_library_sources(phylanx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/ir/*.cpp"
  APPEND)
add_phylanx_library_sources(phylanx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/performance_counters/*.cpp"
  APPEND)
add_phylanx_library_sources(phylanx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/util/*.cpp"
  APPEND)

add_phylanx_source_group(
  NAME phylanx
  CLASS "Source Files"
  ROOT "${PROJECT_SOURCE_DIR}/src"
  TARGETS ${phylanx_SOURCES})

if(MSVC)
  # libphylanx headers
  add_phylanx_library_headers(phylanx
    GLOB GLOBS "${PROJECT_SOURCE_DIR}/phylanx/*.hpp")
  add_phylanx_library_headers(phylanx
    GLOB GLOBS "${PROJECT_SOURCE_DIR}/phylanx/config/*.hpp"
    APPEND)
  add_phylanx_library_headers(phylanx
    GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/phylanx/ast/*.hpp"
    APPEND)
  add_phylanx_library_headers(phylanx
    GLOB GLOBS "${PROJECT_SOURCE_DIR}/phylanx/ir/*.hpp"
    APPEND)
  add_phylanx_library_headers(phylanx
    GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/phylanx/execution_tree/*.hpp"
    APPEND)
  add_phylanx_library_headers(phylanx
    GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/phylanx/util/*.hpp"
    APPEND)
  add_phylanx_library_headers(phylanx
    GLOB GLOBS "${PROJECT_SOURCE_DIR}/phylanx/include/*.hpp"
    APPEND)

  add_phylanx_source_group(
    NAME phylanx_component
    CLASS "Header Files"
    ROOT "${PROJECT_SOURCE_DIR}/phylanx"
    TARGETS ${phylanx_HEADERS})

  # add natvis files to solution (supported starting VS2015)
  if(MSVC14)
    add_phylanx_library_sources(natvis_files
      GLOB GLOBS "${PROJECT_SOURCE_DIR}/tools/VS/*.natvis")
    set(phylanx_external_OBJECTS
        ${phylanx_external_OBJECTS}
        ${natvis_files_SOURCES})
    source_group("Natvis Files" FILES ${natvis_files_SOURCES})
  endif()

endif()

################################################################################
# create main module
set(HPX_TLL_PUBLIC "PUBLIC")
if(phylanx_external_OBJECTS)
  add_hpx_component(phylanx SHARED
    SOURCES ${phylanx_SOURCES}
    HEADERS ${phylanx_HEADERS}
    AUXILIARY ${phylanx_external_OBJECTS}
    COMPONENT_DEPENDENCIES iostreams)
else()
  add_hpx_component(phylanx SHARED
    SOURCES ${phylanx_SOURCES}
    HEADERS ${phylanx_HEADERS}
    COMPONENT_DEPENDENCIES iostreams)
endif()

target_link_libraries(phylanx_component
  PUBLIC
    ${HPX_LIBRARIES}
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES})
target_link_libraries(phylanx_component
  PRIVATE
    blaze::blaze)

if(PHYLANX_WITH_HIGHFIVE)
  target_link_libraries(phylanx_component
    PRIVATE
      HighFive
      ${PHYLANX_HDF5_LIBRARIES})
endif()

set_target_properties(
  phylanx_component PROPERTIES
    VERSION ${PHYLANX_VERSION}
    SOVERSION ${PHYLANX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1
    FOLDER "Core")

set_property(TARGET phylanx_component APPEND
  PROPERTY COMPILE_DEFINITIONS
  "PHYLANX_EXPORTS")

set(phylanx_targets ${phylanx_targets} phylanx_component)

get_property(PHYLANX_TARGET_COMPILE_OPTIONS_VAR
  GLOBAL PROPERTY PHYLANX_TARGET_COMPILE_OPTIONS)
foreach(_flag ${PHYLANX_TARGET_COMPILE_OPTIONS_VAR})
  target_compile_options(phylanx_component INTERFACE ${_flag})
  target_compile_options(phylanx_component PUBLIC ${_flag})
endforeach()

install(
  TARGETS ${phylanx_targets}
  EXPORT PhylanxTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  COMPONENT runtime
  OPTIONAL)

phylanx_export_targets(${phylanx_targets})
