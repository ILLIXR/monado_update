# Find glslc compiler
message("STARTING GLSL")
find_program(GLSLC_EXECUTABLE
        NAMES glslc
        HINTS
        "$ENV{VULKAN_SDK}/Bin"
        "$ENV{VULKAN_SDK}/bin"
        DOC "glslc shader compiler"
)

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Please install Vulkan SDK.")
endif()

message(STATUS "Found glslc: ${GLSLC_EXECUTABLE}")

# Find Python for embedding (Windows)
find_package(Python3 COMPONENTS Interpreter)

# =============================================================================
# APPROACH 2: Embedded SPIR-V (compiled into binary)
# =============================================================================
function(embed_spirv_shader TARGET SHADER_SOURCE)
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_SOURCE} EXT)

    # Determine shader stage from extension
    if(SHADER_EXT STREQUAL ".comp")
        set(SHADER_STAGE "compute")
    elseif(SHADER_EXT STREQUAL ".vert")
        set(SHADER_STAGE "vertex")
    elseif(SHADER_EXT STREQUAL ".frag")
        set(SHADER_STAGE "fragment")
    else()
        message(FATAL_ERROR "Unknown shader extension: ${SHADER_EXT}")
    endif()

    # Output paths
    set(SPIRV_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}${SHADER_EXT}.spv")
    set(HEADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME}_spirv.h")

    # Get absolute path to shader source
    set(SHADER_ABS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SOURCE}")

    # Create output directory
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/shaders")

    # Step 1: Compile GLSL to SPIR-V
    add_custom_command(
            OUTPUT ${SPIRV_OUTPUT}
            COMMAND ${GLSLC_EXECUTABLE}
            -fshader-stage=${SHADER_STAGE}
            -O
            -g
            -o ${SPIRV_OUTPUT}
            ${SHADER_ABS_PATH}
            DEPENDS ${SHADER_ABS_PATH}
            COMMENT "Compiling shader: ${SHADER_NAME}${SHADER_EXT}"
            VERBATIM
    )

    # Step 2: Convert SPIR-V to C header
    if(UNIX)
        # Use xxd on Unix/Linux
        add_custom_command(
                OUTPUT ${HEADER_OUTPUT}
                COMMAND xxd -i ${SPIRV_OUTPUT} > ${HEADER_OUTPUT}
                DEPENDS ${SPIRV_OUTPUT}
                COMMENT "Embedding shader: ${SHADER_NAME}${SHADER_EXT}.spv"
                VERBATIM
        )
    elseif(WIN32 AND Python3_FOUND)
        # Use Python script on Windows
        add_custom_command(
                OUTPUT ${HEADER_OUTPUT}
                COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../scripts/bin2header.py
                ${SPIRV_OUTPUT}
                ${HEADER_OUTPUT}
                ${SHADER_NAME}_spirv
                DEPENDS ${SPIRV_OUTPUT} ${CMAKE_CURRENT_SOURCE_DIR}/../../../scripts/bin2header.py
                COMMENT "Embedding shader: ${SHADER_NAME}${SHADER_EXT}.spv"
                VERBATIM
        )
    else()
        message(FATAL_ERROR "Cannot embed shaders: xxd not available on Windows and Python3 not found")
    endif()

    # Add generated header to target
    target_sources(${TARGET} PRIVATE ${HEADER_OUTPUT})

    # Make sure generated headers are in include path
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    message(STATUS "Shader ${SHADER_NAME}${SHADER_EXT} will be embedded in ${TARGET}")
endfunction()