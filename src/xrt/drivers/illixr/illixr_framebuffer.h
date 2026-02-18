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

	// Depth attachment for rendering
	VkImage depth_attachment_image;
	VkDeviceMemory depth_attachment_memory;
	VkImageView depth_attachment_view;
	VkDeviceSize depth_attachment_size;
	VkDeviceSize depth_attachment_offset;
	VkExtent2D depth_attachment_extent;

	VkFramebuffer handle;
};

#define OFFLOAD_BUFFER_POOL_SIZE 6
