include_guard(GLOBAL)

function(flash_attention_configure_mha_sources)
  set(options)
  set(one_value_args SOURCE_ROOT OUTPUT_ROOT HIPIFY_PERL OUT_HOST OUT_KERNELS OUT_HEADERS OUT_TARGET)
  cmake_parse_arguments(MHA "${options}" "${one_value_args}" "" ${ARGN})

  foreach(required_argument SOURCE_ROOT OUTPUT_ROOT HIPIFY_PERL OUT_HOST OUT_KERNELS OUT_HEADERS OUT_TARGET)
    if(NOT MHA_${required_argument})
      message(FATAL_ERROR "flash_attention_configure_mha_sources: ${required_argument} is required")
    endif()
  endforeach()

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(generator "${CMAKE_CURRENT_SOURCE_DIR}/tools/hipify_mha.py")
  set(snapshot "${MHA_OUTPUT_ROOT}/resolved_sources.txt")
  file(MAKE_DIRECTORY "${MHA_OUTPUT_ROOT}")

  # CONFIGURE_DEPENDS makes adding or removing a forward translation unit
  # re-run CMake. The resolved include closure below adds existing local
  # headers to CMAKE_CONFIGURE_DEPENDS, so a new include is discovered when
  # its including file changes.
  file(GLOB forward_inputs CONFIGURE_DEPENDS
    "${MHA_SOURCE_ROOT}/src/flash_fwd*.cu")
  if(NOT forward_inputs)
    message(FATAL_ERROR "No forward FlashAttention sources were discovered")
  endif()

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${generator}"
      --source-root "${MHA_SOURCE_ROOT}"
      --check-sources
      --snapshot "${snapshot}"
    RESULT_VARIABLE discovery_result
    OUTPUT_VARIABLE discovery_output
    ERROR_VARIABLE discovery_error
  )
  if(NOT discovery_result EQUAL 0)
    message(FATAL_ERROR
      "Pure HIP MHA source discovery failed:\n${discovery_output}${discovery_error}")
  endif()

  file(STRINGS "${snapshot}" source_lines REGEX "^(host|header|kernel) ")
  set(input_files)
  set(generated_host)
  set(generated_headers)
  set(generated_kernels)
  set(generated_outputs)
  foreach(source_line IN LISTS source_lines)
    string(REGEX MATCH "^([^ ]+) (.+)$" _ "${source_line}")
    set(source_kind "${CMAKE_MATCH_1}")
    set(relative_path "${CMAKE_MATCH_2}")
    list(APPEND input_files "${MHA_SOURCE_ROOT}/${relative_path}")

    if(source_kind STREQUAL "host")
      get_filename_component(source_directory "${relative_path}" DIRECTORY)
      get_filename_component(source_stem "${relative_path}" NAME_WE)
      if(source_directory)
        set(generated_path "${source_directory}/${source_stem}_hip.cpp")
      else()
        set(generated_path "${source_stem}_hip.cpp")
      endif()
      set(generated_host "${MHA_OUTPUT_ROOT}/${generated_path}")
    elseif(source_kind STREQUAL "kernel")
      string(REGEX REPLACE "\\.cu$" ".hip" generated_path "${relative_path}")
      list(APPEND generated_kernels "${MHA_OUTPUT_ROOT}/${generated_path}")
    elseif(source_kind STREQUAL "header")
      set(generated_path "${relative_path}")
      list(APPEND generated_headers "${MHA_OUTPUT_ROOT}/${generated_path}")
    else()
      message(FATAL_ERROR "Unknown source kind in ${snapshot}: ${source_kind}")
    endif()
    list(APPEND generated_outputs "${MHA_OUTPUT_ROOT}/${generated_path}")
  endforeach()

  if(NOT generated_host OR NOT generated_kernels OR NOT generated_headers)
    message(FATAL_ERROR "Incomplete pure HIP MHA source closure in ${snapshot}")
  endif()

  set(runtime_header "${MHA_OUTPUT_ROOT}/src/flash_mha_runtime.h")
  set(dtk_macros_header "${MHA_OUTPUT_ROOT}/src/flash_dtk_macros.h")
  list(APPEND generated_headers "${runtime_header}")
  list(APPEND generated_headers "${dtk_macros_header}")
  list(APPEND generated_outputs "${runtime_header}")
  list(APPEND generated_outputs "${dtk_macros_header}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${generator}" ${input_files})

  set(generation_stamp "${MHA_OUTPUT_ROOT}/generation.stamp")
  add_custom_command(
    OUTPUT "${generation_stamp}"
    BYPRODUCTS ${generated_outputs}
    COMMAND "${Python3_EXECUTABLE}" "${generator}"
      --source-root "${MHA_SOURCE_ROOT}"
      --generate
      --output-root "${MHA_OUTPUT_ROOT}"
      --hipify-perl "${MHA_HIPIFY_PERL}"
      --snapshot "${snapshot}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${generation_stamp}"
    DEPENDS "${generator}" ${input_files}
    COMMENT "Generating pure HIP FlashAttention MHA sources"
    VERBATIM
  )
  add_custom_target(mha_hip_codegen DEPENDS "${generation_stamp}")

  set_source_files_properties(${generated_outputs} PROPERTIES GENERATED TRUE)
  set(${MHA_OUT_HOST} "${generated_host}" PARENT_SCOPE)
  set(${MHA_OUT_KERNELS} "${generated_kernels}" PARENT_SCOPE)
  set(${MHA_OUT_HEADERS} "${generated_headers}" PARENT_SCOPE)
  set(${MHA_OUT_TARGET} mha_hip_codegen PARENT_SCOPE)
endfunction()
