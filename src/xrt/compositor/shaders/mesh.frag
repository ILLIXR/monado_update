// Copyright 2017, James Sarrett.
// Copyright 2017, Bastiaan Olij.
// Copyright 2019, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: James Sarrett <jsarrett@gmail.com>
// Author: Bastiaan Olij <mux213@gmail.com>
// Author: Lubosz Sarnecki <lubosz.sarnecki@collabora.com>
// Author: Pete Black <pete.black@collabora.com>

#version 450
#extension GL_GOOGLE_include_directive : require

#include "srgb.inc.glsl"

layout(constant_id = 0) const bool encode_srgb = false;


layout (binding = 0) uniform sampler2D tex_sampler;

layout (location = 0)  in vec2 in_ruv;
layout (location = 1)  in vec2 in_guv;
layout (location = 2)  in vec2 in_buv;
layout (location = 0) out vec4 out_color;


void main()
{
	float r = texture(tex_sampler, in_ruv).x;
	float g = texture(tex_sampler, in_guv).y;
	float b = texture(tex_sampler, in_buv).z;

	vec3 color = vec3(r, g, b);
	if (encode_srgb) {
		color = from_linear_to_srgb(color);
	}
	out_color = vec4(color, 1.0);
}
