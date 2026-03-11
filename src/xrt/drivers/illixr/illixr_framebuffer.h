#pragma once

// Define the framebuffer struct (duplicate from comp_renderer.h, but that's OK)
struct illixr_framebuffer
{
	VkImage image;
	VkDeviceMemory memory;
	VkImageView view;
	VkDeviceSize image_size;
	VkDeviceSize image_offset;
	VkExtent2D image_extent;

	// Depth images for encoding
	VkImage depth_image;
	VkDeviceMemory depth_memory;
	VkImageView depth_view;
	VkDeviceSize depth_size;
	VkDeviceSize depth_offset;
	VkExtent2D depth_extent;

	// Projection clip planes from XrCompositionLayerDepthInfoKHR.
	// near_z is the near clip distance (smaller positive value) and far_z is
	// the far clip distance.  The encoder passes these to the decoder so it can
	// linearise the encoded depth values back into view-space metres.
	// Defaults match typical VR usage; overwritten whenever a depth layer is present.
	float near_z;
	float far_z;

	// Depth attachment for rendering
	VkImage depth_attachment_image;
	VkDeviceMemory depth_attachment_memory;
	VkImageView depth_attachment_view;
	VkDeviceSize depth_attachment_size;
	VkDeviceSize depth_attachment_offset;
	VkExtent2D depth_attachment_extent;

	// Motion vector image from Unity quad layer (RGBA16F, RG = NDC delta XY)
	VkImage motion_vec_image;
	VkDeviceMemory motion_vec_memory;
	VkImageView motion_vec_view;
	VkDeviceSize motion_vec_size;
	VkDeviceSize motion_vec_offset;
	VkExtent2D motion_vec_extent;
	VkFramebuffer handle;
};

#define OFFLOAD_BUFFER_POOL_SIZE 6
