// Copyright 2020-2026, The Board of Trustees of the University of Illinois.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ILLIXR shared memory for Quest 3 motion vectors
 * @author RSIM Group <illixr@cs.illinois.edu>
 * @ingroup drv_illixr
 */

// illixr_mv_shmem.h
//
// Win32 named shared memory layout shared between IllixrXrHook.dll (Unity
// process) and the Monado compositor (comp_renderer.c / comp_compositor.c).
//
// The hook DLL writes ring_index and frame_seq after each successful
// xrReleaseSwapchainImage for the motion vector swapchain.  The Monado
// compositor reads them in dispatch_graphics to select which VkImage to
// blit from.
//
// The GPU sync guarantee is provided by xrReleaseSwapchainImage itself:
// Monado's D3D11/Vulkan interop layer signals a shared fence when the image
// is released, and the compositor waits on that fence before reading the
// image.  The CPU-side shared memory only carries the ring buffer index and
// sequence number — no GPU data passes through it.
//
// Compatible with both MSVC (hook DLL, C++) and GCC/Clang (Monado, C).

#pragma once

#ifdef _WIN32
#define ILLIXR_MV_SHMEM_NAME "Local\\ILLIXR_MotionVectors_v1"
#else
// POSIX fallback name (not currently used — both processes are Windows)
#define ILLIXR_MV_SHMEM_NAME "/ILLIXR_MotionVectors_v1"
#endif

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

// Size of the shared memory mapping (bytes).  Fixed at 64 for alignment.
#define ILLIXR_MV_SHMEM_SIZE 64

// Packed to ensure identical layout between MSVC and GCC.
#pragma pack(push, 1)
typedef struct
{
	// Written by hook DLL, read by Monado compositor.
	//
	// ring_index: the swapchain image index last written by the DLL.
	// This is the index returned by xrAcquireSwapchainImage for the motion
	// vector swapchain, captured immediately before xrReleaseSwapchainImage.
	//
	// frame_seq: incremented by 1 each time ring_index is updated.
	// Monado compares this to the sequence number it saw on the previous
	// frame; if equal, the index is stale (Unity hasn't submitted a new
	// frame yet) and the compositor skips the blit to avoid re-using data
	// that may be in the middle of being written.
	volatile uint32_t ring_index;
	volatile uint32_t frame_seq;

	// Reserved for future use.  Must be zero.
	volatile uint32_t reserved[14];
} IllixrMvShmem;
#pragma pack(pop)

// Compile-time size check.
#ifdef __cplusplus
static_assert(sizeof(IllixrMvShmem) == ILLIXR_MV_SHMEM_SIZE, "IllixrMvShmem size mismatch");
#endif
