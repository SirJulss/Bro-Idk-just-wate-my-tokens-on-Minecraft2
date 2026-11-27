# GPU capability report

| Property             | Value                                                       |
|----------------------|-------------------------------------------------------------|
| Adapter              | NVIDIA GeForce GTX 1080                                     |
| Backend              | Vulkan                                                      |
| Device type          | discrete GPU                                                |
| Vendor / device id   | 0x10DE / 0x1B80                                             |
| Driver               | NVIDIA                                                      |
| Driver info          | 582.53                                                      |
| PCI bus id           | 0000:01:00.0                                                |
| Quality tier         | high - discrete GPU with indirect bucket draws: all effects |
| Draw path            | A: GPU culling, multi_draw_indexed_indirect_count           |
| Instance flags       | VALIDATION_INDIRECT_CALL                                    |
| DX12 shader compiler | fxc                                                         |
| WebGPU compliant     | yes                                                         |

## Derived capabilities

| Capability                                             | Value | Meaning                                                                                      |
|--------------------------------------------------------|-------|----------------------------------------------------------------------------------------------|
| can_multi_draw_indirect_count                          | yes   | multi_draw_(indexed_)indirect_count may be called                                            |
| can_indirect_first_instance                            | yes   | first_instance of indirect draws offsets instance-step attributes (bucket)                   |
| indirect_draws_respect_first_vertex_and_instance       | yes   | multi_draw_indirect: vertex_index and instance_index include first_vertex and first_instance |
| indirect_count_draws_respect_first_vertex_and_instance | yes   | the same for multi_draw_indirect_count                                                       |
| can_timestamp_passes                                   | yes   | timestamps at the begin and end of a pass                                                    |
| can_timestamp_inside_encoders                          | yes   | CommandEncoder::write_timestamp between passes                                               |
| can_timestamp_inside_passes                            | yes   | write_timestamp inside render and compute passes                                             |
| hdr_rg11b10_renderable                                 | yes   | Rg11b10Ufloat as render target for HDR intermediates                                         |
| float32_filterable                                     | yes   | linear filtering of R32Float, Rg32Float and Rgba32Float                                      |
| norm16_textures                                        | yes   | R16Unorm and the other 16-bit normalised formats                                             |
| bc_compressed_textures                                 | yes   | BC1 to BC7 texture formats                                                                   |
| can_unclip_depth                                       | yes   | PrimitiveState::unclipped_depth (depth clamp)                                                |
| can_dual_source_blend                                  | yes   | dual-source blend factors                                                                    |
| can_use_immediates                                     | yes   | immediate data in pipeline layouts (var<immediate>)                                          |
| adapter_format_features                                | yes   | adapter-specific texture format capabilities                                                 |

## Optional features

13 of the 13 optional features the pipeline can use are enabled.

| Feature                                  | Available | Enabled | Used for                                                                        | Without it                                                                      |
|------------------------------------------|-----------|---------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| MULTI_DRAW_INDIRECT_COUNT                | yes       | yes     | draw path A (Vulkan): GPU culling writes the indirect arguments and their count | draw path B: culling on the CPU, multi_draw_indexed_indirect with a known count |
| INDIRECT_FIRST_INSTANCE                  | yes       | yes     | first_instance of indirect draws selects the bucket of a draw (both draw paths) | draw path B with one direct draw per visible range                              |
| TIMESTAMP_QUERY                          | yes       | yes     | GPU time per pass (timestamps at pass begin and end)                            | no GPU timings, CPU frame times only                                            |
| TIMESTAMP_QUERY_INSIDE_ENCODERS          | yes       | yes     | timestamps between passes (uploads, whole frame)                                | pass timings only                                                               |
| TIMESTAMP_QUERY_INSIDE_PASSES            | yes       | yes     | timing scopes inside a render or compute pass                                   | one timing per pass                                                             |
| RG11B10UFLOAT_RENDERABLE                 | yes       | yes     | Rg11b10Ufloat targets without alpha: bloom, half-resolution reflections         | Rgba16Float targets (8 instead of 4 bytes per pixel)                            |
| FLOAT32_FILTERABLE                       | yes       | yes     | linear sampling of 32-bit float textures (height maps, depth pyramid)           | nearest sampling with interpolation in the shader                               |
| TEXTURE_FORMAT_16BIT_NORM                | yes       | yes     | macro height maps in R16Unorm (2 bytes per texel)                               | macro height maps in R32Float (4 bytes per texel)                               |
| TEXTURE_COMPRESSION_BC                   | yes       | yes     | BC-compressed textures for large maps                                           | uncompressed textures                                                           |
| DEPTH_CLIP_CONTROL                       | yes       | yes     | unclipped depth for shadow cascades (casters in front of the near plane)        | longer cascade depth ranges                                                     |
| DUAL_SOURCE_BLENDING                     | yes       | yes     | per-channel transmittance when blending atmosphere, fog and clouds              | single-channel transmittance                                                    |
| IMMEDIATES                               | yes       | yes     | small per-pass constants as immediate data, at most 128 bytes (no bind group)   | a small uniform buffer                                                          |
| TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES | yes       | yes     | format capabilities beyond WebGPU, e.g. read-write storage textures             | the format capabilities WebGPU guarantees                                       |

Other features of the adapter (not requested): SHADER_FLOAT32_ATOMIC, PIPELINE_STATISTICS_QUERY, MAPPABLE_PRIMARY_BUFFERS, TEXTURE_BINDING_ARRAY, BUFFER_BINDING_ARRAY, STORAGE_RESOURCE_BINDING_ARRAY, SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING, STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING, PARTIALLY_BOUND_BINDING_ARRAY, ADDRESS_MODE_CLAMP_TO_ZERO, ADDRESS_MODE_CLAMP_TO_BORDER, POLYGON_MODE_LINE, POLYGON_MODE_POINT, CONSERVATIVE_RASTERIZATION, VERTEX_WRITABLE_STORAGE, CLEAR_TEXTURE, MULTIVIEW, TEXTURE_ATOMIC, TEXTURE_FORMAT_NV12, TEXTURE_FORMAT_P010, SHADER_F64, SHADER_I16, SHADER_EARLY_DEPTH_TEST, SHADER_INT64, SUBGROUP, SUBGROUP_VERTEX, SUBGROUP_BARRIER, PIPELINE_CACHE, SHADER_INT64_ATOMIC_MIN_MAX, SHADER_INT64_ATOMIC_ALL_OPS, VULKAN_EXTERNAL_MEMORY_WIN32, TEXTURE_INT64_ATOMIC, EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS, PASSTHROUGH_SHADERS, SELECTIVE_MULTIVIEW, MULTISAMPLE_ARRAY, SHADER_DRAW_INDEX, MEMORY_DECORATION_COHERENT, MEMORY_DECORATION_VOLATILE, EXPERIMENTAL_RAY_TRACING_PIPELINES, DEPTH32FLOAT_STENCIL8, TEXTURE_COMPRESSION_BC_SLICED_3D, BGRA8UNORM_STORAGE, FLOAT32_BLENDABLE, CLIP_DISTANCES, PRIMITIVE_INDEX

## Limits

Requested is what the pipeline asks for; enabled is that value clamped to the adapter.

| Limit                                                    | WebGPU default | Requested  | Adapter                 | Enabled | Used for                                            |
|----------------------------------------------------------|----------------|------------|-------------------------|---------|-----------------------------------------------------|
| max_buffer_size                                          | 256 MiB        | 1 GiB      | 4292870144 B (~4.0 GiB) | 1 GiB   | largest buffer: the quad pool (never above 4 GiB)   |
| max_storage_buffer_binding_size                          | 128 MiB        | 1 GiB      | 2147483644 B (~2.0 GiB) | 1 GiB   | the quad pool bound as one storage buffer           |
| max_texture_array_layers                                 | 256            | 4294967295 | 2048                    | 2048    | block texture arrays                                |
| max_immediate_size                                       | 0 B            | 128 B      | 256 B                   | 128 B   | immediate data per pass                             |
| max_sampled_textures_per_shader_stage                    | 16             | 32         | 524288                  | 32      | inputs of the lighting pass                         |
| max_storage_textures_per_shader_stage                    | 4              | 16         | 524288                  | 16      | single-mip views of the depth pyramid               |
| max_color_attachment_bytes_per_sample                    | 32 B           | 64 B       | 128 B                   | 64 B    | G-buffer bytes per pixel                            |
| max_bind_groups                                          | 4              | 8          | 8                       | 8       | frame, pass, material and object bind groups        |
| max_storage_buffers_per_shader_stage                     | 8              | 8          | 524288                  | 8       | pools bound by culling and drawing                  |
| max_buffers_and_acceleration_structures_per_shader_stage | 28             | 28         | 4294967295              | 28      | storage + uniform + vertex buffers of one stage     |
| max_uniform_buffer_binding_size                          | 64 KiB         | 64 KiB     | 64 KiB                  | 64 KiB  | frame and pass constants                            |
| max_uniform_buffers_per_shader_stage                     | 12             | 12         | 15                      | 12      | constant blocks of one stage                        |
| max_samplers_per_shader_stage                            | 16             | 16         | 524288                  | 16      | samplers of one stage                               |
| max_bindings_per_bind_group                              | 1000           | 1000       | 4294967295              | 1000    | highest binding index                               |
| max_texture_dimension_2d                                 | 8192           | 8192       | 32768                   | 8192    | render targets, shadow atlas, macro maps            |
| max_texture_dimension_3d                                 | 2048           | 2048       | 16384                   | 2048    | light volumes, froxel grids, cloud noise            |
| max_color_attachments                                    | 8              | 8          | 8                       | 8       | G-buffer targets                                    |
| max_compute_invocations_per_workgroup                    | 256            | 256        | 1536                    | 256     | compute workgroup size                              |
| max_compute_workgroups_per_dimension                     | 65535          | 65535      | 65535                   | 65535   | culling dispatch size                               |
| min_uniform_buffer_offset_alignment                      | 256 B          | 256 B      | 256 B                   | 256 B   | uniform offsets in shared buffers (lower is better) |
| min_storage_buffer_offset_alignment                      | 256 B          | 256 B      | 32 B                    | 256 B   | storage offsets in shared buffers (lower is better) |

## Downlevel flags

Shader model: Sm5. Missing flags: none

| Flag                                 | Present |
|--------------------------------------|---------|
| COMPUTE_SHADERS                      | yes     |
| FRAGMENT_WRITABLE_STORAGE            | yes     |
| INDIRECT_EXECUTION                   | yes     |
| BASE_VERTEX                          | yes     |
| READ_ONLY_DEPTH_STENCIL              | yes     |
| NON_POWER_OF_TWO_MIPMAPPED_TEXTURES  | yes     |
| CUBE_ARRAY_TEXTURES                  | yes     |
| COMPARISON_SAMPLERS                  | yes     |
| INDEPENDENT_BLEND                    | yes     |
| VERTEX_STORAGE                       | yes     |
| ANISOTROPIC_FILTERING                | yes     |
| FRAGMENT_STORAGE                     | yes     |
| MULTISAMPLED_SHADING                 | yes     |
| DEPTH_TEXTURE_AND_BUFFER_COPIES      | yes     |
| WEBGPU_TEXTURE_FORMAT_SUPPORT        | yes     |
| BUFFER_BINDINGS_NOT_16_BYTE_ALIGNED  | yes     |
| UNRESTRICTED_INDEX_BUFFER            | yes     |
| FULL_DRAW_INDEX_UINT32               | yes     |
| DEPTH_BIAS_CLAMP                     | yes     |
| VIEW_FORMATS                         | yes     |
| UNRESTRICTED_EXTERNAL_TEXTURE_COPIES | yes     |
| SURFACE_VIEW_FORMATS                 | yes     |
| NONBLOCKING_QUERY_RESOLVE            | yes     |
| SHADER_F16_IN_F32                    | yes     |
| MSL2_1                               | yes     |
| TEXTURE_COMPRESSION                  | yes     |
