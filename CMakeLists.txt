# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(MLAS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(ONNXRUNTIME_MLAS_MULTI_ARCH FALSE)
add_library(onnxruntime_mlas
  ${MLAS_SRC_DIR}/platform.cpp
  ${MLAS_SRC_DIR}/threading.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
  ${MLAS_SRC_DIR}/qgemm.cpp
  ${MLAS_SRC_DIR}/qdwconv.cpp
  ${MLAS_SRC_DIR}/convolve.cpp
  ${MLAS_SRC_DIR}/convsym.cpp
  ${MLAS_SRC_DIR}/pooling.cpp
  ${MLAS_SRC_DIR}/transpose.cpp
  ${MLAS_SRC_DIR}/reorder.cpp
  ${MLAS_SRC_DIR}/snchwc.cpp
  ${MLAS_SRC_DIR}/activate.cpp
  ${MLAS_SRC_DIR}/logistic.cpp
  ${MLAS_SRC_DIR}/tanh.cpp
  ${MLAS_SRC_DIR}/erf.cpp
  ${MLAS_SRC_DIR}/compute.cpp
  ${MLAS_SRC_DIR}/quantize.cpp
  ${MLAS_SRC_DIR}/qgemm_kernel_default.cpp
  ${MLAS_SRC_DIR}/qladd.cpp
  ${MLAS_SRC_DIR}/qlmul.cpp
  ${MLAS_SRC_DIR}/qpostprocessor.cpp
  ${MLAS_SRC_DIR}/qlgavgpool.cpp
  ${MLAS_SRC_DIR}/qdwconv_kernelsize.cpp
)

set(ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas)


# if(TRUE)
#     #Linux/FreeBSD/PowerPC/...
#     #The value of CMAKE_SYSTEM_PROCESSOR should be from `uname -m`
#     #Example values:
#     #arm64v8/ubuntu -> aarch64
#     #arm32v6/alpine -> armv7l
#     #arm32v7/centos -> armv7l
#     #ppc64le/debian -> ppc64le
#     #s390x/ubuntu -> s390x
#     #ppc64le/busybox -> ppc64le
#     #arm64v8/ubuntu -> aarch64
#     #Android: armv7-a aarch64 i686 x86_64
#     #chasun: I don't think anyone uses 'arm64'
#     if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
#         set(ARM64 TRUE)
#     elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
#         set(ARM TRUE)
#     elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
#         set(ARM64 TRUE)
#     elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc.*|ppc.*)")
#         set(POWER TRUE)
#     elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
#         set(X86 TRUE)
#     elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
#         set(X86_64 TRUE)
#     endif()
# endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
    set(X86 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set(X86_64 TRUE)
endif()

#If ONNXRUNTIME_MLAS_MULTI_ARCH is true, we need to go through every if branch below
#and split MLAS to multiple static libraries.
#Otherwise, it works like if(...) elseif(...) elseif(...) endif()
set(MLAS_SOURCE_IS_NOT_SET 1)

if(X86 AND MLAS_SOURCE_IS_NOT_SET)
    enable_language(ASM)

    set(mlas_platform_srcs_sse2
        ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
        ${MLAS_SRC_DIR}/x86/SgemmKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
        ${MLAS_SRC_DIR}/x86/SgemmKernelAvx.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs
        ${mlas_platform_srcs_sse2}
        ${mlas_platform_srcs_avx}
    )
    if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
        set(MLAS_SOURCE_IS_NOT_SET 0)
    endif()
endif()
if(X86_64 AND MLAS_SOURCE_IS_NOT_SET)
    message("!!!!!!!!!@$$$$$$")
    enable_language(ASM)

    # Forward the flags for the minimum target platform version from the C
    # compiler to the assembler. This works around CMakeASMCompiler.cmake.in
    # not including the logic to set this flag for the assembler.
    set(CMAKE_ASM${ASM_DIALECT}_OSX_DEPLOYMENT_TARGET_FLAG "${CMAKE_C_OSX_DEPLOYMENT_TARGET_FLAG}")

    # The LLVM assembler does not support the .arch directive to enable instruction
    # set extensions and also doesn't support AVX-512F instructions without
    # turning on support via command-line option. Group the sources by the
    # instruction set extension and explicitly set the compiler flag as appropriate.

    set(mlas_platform_srcs_sse2
        ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
        ${MLAS_SRC_DIR}/x86_64/DgemmKernelSse2.S
        ${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
        ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
        ${MLAS_SRC_DIR}/x86_64/SconvKernelSse2.S
        ${MLAS_SRC_DIR}/x86_64/SpoolKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
        ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx.S
        ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
        ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
        ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
        ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
        ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx.S
        ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx.S
        ${MLAS_SRC_DIR}/x86_64/SoftmaxKernelAvx.S
        ${MLAS_SRC_DIR}/intrinsics/avx/min_max_elements.cpp
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs_avx2
        ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAvx2.S
        ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx2.S
        ${MLAS_SRC_DIR}/x86_64/QgemmU8U8KernelAvx2.S
        ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvxVnni.S
        ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx2.S
        ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx2.S
        ${MLAS_SRC_DIR}/x86_64/DgemmKernelFma3.S
        ${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
        ${MLAS_SRC_DIR}/x86_64/SconvKernelFma3.S
        ${MLAS_SRC_DIR}/x86_64/TransKernelFma3.S
        ${MLAS_SRC_DIR}/x86_64/LogisticKernelFma3.S
        ${MLAS_SRC_DIR}/x86_64/TanhKernelFma3.S
        ${MLAS_SRC_DIR}/x86_64/ErfKernelFma3.S
        ${MLAS_SRC_DIR}/intrinsics/avx2/qladd_avx2.cpp
        ${MLAS_SRC_DIR}/intrinsics/avx2/qdwconv_avx2.cpp
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

    set(mlas_platform_srcs_avx512f
        ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx512F.S
        ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx512F.S
        ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx512F.S
        ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx512F.S
        ${MLAS_SRC_DIR}/x86_64/TransKernelAvx512F.S
        ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
    )
    set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

    set(mlas_platform_srcs_avx512core
        ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Core.S
        ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Vnni.S
        ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx512Core.S
        ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx512Core.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx512core} PROPERTIES COMPILE_FLAGS "-mavx512bw -mavx512dq -mavx512vl")

    set(mlas_platform_srcs
        ${MLAS_SRC_DIR}/dgemm.cpp
        ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
        ${mlas_platform_srcs_sse2}
        ${mlas_platform_srcs_avx}
        ${mlas_platform_srcs_avx2}
        ${mlas_platform_srcs_avx512f}
        ${mlas_platform_srcs_avx512core}
    )
    message("${mlas_platform_srcs}")
    if(ONNXRUNTIME_MLAS_MULTI_ARCH)
        onnxruntime_add_static_library(onnxruntime_mlas_x86_64 ${mlas_platform_srcs})
        set_target_properties(onnxruntime_mlas_x86_64 PROPERTIES OSX_ARCHITECTURES "x86_64")
        list(APPEND ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas_x86_64)
        set(mlas_platform_srcs )
    else()
        set(MLAS_SOURCE_IS_NOT_SET 0)
    endif()
endif()
if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH AND MLAS_SOURCE_IS_NOT_SET)
    file(GLOB_RECURSE mlas_platform_srcs
        "${MLAS_SRC_DIR}/scalar/*.cpp")
endif()
target_sources(onnxruntime_mlas PRIVATE ${mlas_platform_srcs})

foreach(mlas_target ${ONNXRUNTIME_MLAS_LIBS})
    target_include_directories(${mlas_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${MLAS_SRC_DIR})
endforeach()
set_target_properties(onnxruntime_mlas PROPERTIES FOLDER "ONNXRuntime")
add_compile_definitions(
    BUILD_MLAS_NO_ONNXRUNTIME
  )

