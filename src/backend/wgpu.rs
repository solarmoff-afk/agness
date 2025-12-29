use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::borrow::Cow;
use std::sync::atomic::{AtomicU64, Ordering};

use raw_window_handle::{HasWindowHandle, HasDisplayHandle};
use wgpu;

use crate::*;
use crate::error::BackendError;

// Platform specific logic for Async execution helper
mod platform {
    use std::future::Future;

    // Native implementation blocks the thread because Agness requires sync init
    #[cfg(not(target_arch = "wasm32"))]
    pub fn execute_sync<F: Future>(future: F) -> F::Output {
        pollster::block_on(future)
    }

    // WASM implementation spawns a local future
    // Since blocking main thread is impossible on web the result is deferred
    #[cfg(target_arch = "wasm32")]
    pub fn execute_sync<F: Future>(future: F) -> F::Output 
    where F::Output: Default {
        wasm_bindgen_futures::spawn_local(async move {
            let _ = future.await;
        });
        Default::default()
    }
    
    // Helper to spawn async initialization on WASM
    #[cfg(target_arch = "wasm32")]
    pub fn spawn_init<F>(future: F, on_complete: impl FnOnce(F::Output) + 'static)
    where F: Future + 'static {
        wasm_bindgen_futures::spawn_local(async move {
            let result = future.await;
            on_complete(result);
        });
    }
}

struct Resources {
    buffers: HashMap<BufferId, wgpu::Buffer>,
    textures: HashMap<TextureId, wgpu::Texture>,
    texture_views: HashMap<TextureId, wgpu::TextureView>,
    samplers: HashMap<SamplerId, wgpu::Sampler>,
    pipelines: HashMap<PipelineId, wgpu::RenderPipeline>,
    compute_pipelines: HashMap<PipelineId, wgpu::ComputePipeline>,
    bind_groups: HashMap<BindGroupId, wgpu::BindGroup>,
}

struct InnerContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: Option<wgpu::Surface<'static>>,
    #[allow(dead_code)] // Keep config for resize handling
    config: Option<wgpu::SurfaceConfiguration>,
}

/// Implementation of Agness backend based on WGPU
pub struct WgpuBackend {
    // Context is wrapped in Option to support deferred initialization on WASM
    context: Arc<RwLock<Option<InnerContext>>>,
    resources: Arc<RwLock<Resources>>,
    next_id: AtomicU64,
}

impl WgpuBackend {
    fn gen_id<T: From<ResourceId>>(&self) -> T {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // Ensure ID is never zero
        T::from(ResourceId::new(if id == 0 { 1 } else { id }).unwrap())
    }

    // Helper to access device safely
    // Panics if called before initialization is complete (relevant for WASM startup)
    fn with_context<F, R>(&self, f: F) -> Result<R, BackendError>
    where F: FnOnce(&InnerContext) -> R {
        let guard = self.context.read().unwrap();
        match &*guard {
            Some(ctx) => Ok(f(ctx)),
            None => {
                crate::agnesslog_warn!("Attempt to use Backend before async initialization finished");
                Err(BackendError::InitializationFailed)
            }
        }
    }
}

impl AgnessBackend for WgpuBackend {
    fn new<W>(window: Option<&W>) -> Result<Self, BackendError>
    where
        W: HasWindowHandle + HasDisplayHandle,
        Self: Sized,
    {
        let instance = wgpu::Instance::default();
        
        // Surface creation needs to be safe and handle errors
        let surface = if let Some(win) = window {
             unsafe {
                let target = wgpu::SurfaceTargetUnsafe::from_window(&win)
                    .map_err(|_| BackendError::InitializationFailed)?;
                
                Some(instance.create_surface_unsafe(target)
                    .map_err(|_| BackendError::InitializationFailed)?)
            }
        } else {
            None
        };

        let context_storage = Arc::new(RwLock::new(None));
        let _context_storage_clone = context_storage.clone();

        // Async preparation logic
        let future = async move {
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface.as_ref(),
                force_fallback_adapter: false,
            }).await?;

            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Agness Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            ).await.ok()?;

            let config = if let Some(surf) = &surface {
                let caps = surf.get_capabilities(&adapter);
                let config = wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: caps.formats[0],
                    width: 800, // Initial size, should be updated on resize
                    height: 600,
                    present_mode: wgpu::PresentMode::Fifo,
                    alpha_mode: caps.alpha_modes[0],
                    view_formats: vec![],
                    desired_maximum_frame_latency: 2,
                };
                surf.configure(&device, &config);
                Some(config)
            } else {
                None
            };

            Some(InnerContext { device, queue, surface, config })
        };

        // Platform specific execution
        #[cfg(not(target_arch = "wasm32"))]
        {
            let ctx = platform::execute_sync(future);
            *context_storage.write().unwrap() = ctx;
        }

        #[cfg(target_arch = "wasm32")]
        {
            // On WASM we spawn and populate the lock later
            platform::spawn_init(future, move |ctx| {
                if let Some(c) = ctx {
                    *context_storage_clone.write().unwrap() = Some(c);
                    crate::agnesslog_verbose!("WASM Initialization Complete");
                }
            });
        }

        Ok(Self {
            context: context_storage,
            resources: Arc::new(RwLock::new(Resources {
                buffers: HashMap::new(),
                textures: HashMap::new(),
                texture_views: HashMap::new(),
                samplers: HashMap::new(),
                pipelines: HashMap::new(),
                compute_pipelines: HashMap::new(),
                bind_groups: HashMap::new(),
            })),
            next_id: AtomicU64::new(1),
        })
    }

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferId, BackendError> {
        self.with_context(|ctx| {
            let usage = map_buffer_usage(desc.usage);
            let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: desc.label.as_deref(),
                size: desc.size,
                usage,
                mapped_at_creation: false,
            });
            
            let id = self.gen_id();
            self.resources.write().unwrap().buffers.insert(id, buffer);
            id
        })
    }

    fn write_buffer(&self, buffer_id: BufferId, offset: u64, data: &[u8]) -> Result<(), BackendError> {
        self.with_context(|ctx| {
            let res = self.resources.read().unwrap();
            if let Some(buf) = res.buffers.get(&buffer_id) {
                ctx.queue.write_buffer(buf, offset, data);
            }
        })
    }

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureId, BackendError> {
        self.with_context(|ctx| {
            let size = wgpu::Extent3d {
                width: desc.width,
                height: desc.height,
                depth_or_array_layers: desc.depth,
            };
            
            let dim = match desc.ty {
                TextureType::TwoDimensions => wgpu::TextureDimension::D2,
                TextureType::ThreeDimensions => wgpu::TextureDimension::D3,
                TextureType::Cube | TextureType::Array => wgpu::TextureDimension::D2,
            };

            let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: desc.label.as_deref(),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: dim,
                format: map_texture_format(desc.format),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST |
                       if desc.render_target { wgpu::TextureUsages::RENDER_ATTACHMENT } else { wgpu::TextureUsages::empty() },
                view_formats: &[],
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            
            let id = self.gen_id();
            let mut res = self.resources.write().unwrap();
            res.textures.insert(id, texture);
            res.texture_views.insert(id, view);
            id
        })
    }

    fn write_texture(&self, texture: TextureId, data: &[u8], bytes_per_row: u32) -> Result<(), BackendError> {
        self.with_context(|ctx| {
            let res = self.resources.read().unwrap();
            if let Some(tex) = res.textures.get(&texture) {
                let size = wgpu::Extent3d {
                    width: tex.width(),
                    height: tex.height(),
                    depth_or_array_layers: tex.depth_or_array_layers(),
                };
                
                ctx.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: None,
                    },
                    size,
                );
            }
        })
    }

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerId, BackendError> {
        self.with_context(|ctx| {
            let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
                address_mode_u: map_address_mode(desc.address_u),
                address_mode_v: map_address_mode(desc.address_v),
                address_mode_w: map_address_mode(desc.address_w),
                mag_filter: map_filter_mode(desc.mag_filter),
                min_filter: map_filter_mode(desc.min_filter),
                anisotropy_clamp: desc.anisotropy_clamp,
                ..Default::default()
            });
            let id = self.gen_id();
            self.resources.write().unwrap().samplers.insert(id, sampler);
            id
        })
    }

    fn create_render_pipeline(&self, desc: RenderPipelineDesc) -> Result<PipelineId, BackendError> {
        self.with_context(|ctx| {
            let vs = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("VS"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(desc.vs_source)),
            });
            let fs = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("FS"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(desc.fs_source)),
            });

            // Logic to support vertex split via layouts mapping
            let mut attributes_storage = Vec::new();
            let mut wgpu_layouts = Vec::new();

            for layout in desc.vertex_layouts {
                let mut attrs = Vec::new();
                for attr in layout.attributes {
                    attrs.push(wgpu::VertexAttribute {
                        format: map_vertex_format(attr.format),
                        offset: attr.offset,
                        shader_location: attr.location,
                    });
                }
                attributes_storage.push(attrs);
            }

            for (i, layout) in desc.vertex_layouts.iter().enumerate() {
                wgpu_layouts.push(wgpu::VertexBufferLayout {
                    array_stride: layout.array_stride,
                    step_mode: match layout.step_mode {
                        VertexStepMode::Vertex => wgpu::VertexStepMode::Vertex,
                        VertexStepMode::Instance => wgpu::VertexStepMode::Instance,
                    },
                    attributes: &attributes_storage[i],
                });
            }

            let layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: desc.label,
                bind_group_layouts: &[], // Auto inferred from shaders
                push_constant_ranges: &[],
            });

            let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: desc.label,
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vs,
                    entry_point: "vs_main",
                    buffers: &wgpu_layouts,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Bgra8Unorm, // Should match swapchain
                        blend: desc.blend.map(map_blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: map_topology(desc.primitive.topology),
                    cull_mode: desc.primitive.cull_mode.map(map_cull),
                    front_face: match desc.primitive.front_face {
                        FrontFace::Ccw => wgpu::FrontFace::Ccw,
                        FrontFace::Cw => wgpu::FrontFace::Cw,
                    },
                    ..Default::default()
                },
                depth_stencil: desc.depth_stencil.map(|ds| wgpu::DepthStencilState {
                    format: map_texture_format(ds.format),
                    depth_write_enabled: ds.depth_write,
                    depth_compare: map_compare(ds.depth_compare),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            let id = self.gen_id();
            self.resources.write().unwrap().pipelines.insert(id, pipeline);
            id
        })
    }

    fn create_compute_pipeline(&self, desc: ComputePipelineDesc) -> Result<PipelineId, BackendError> {
        self.with_context(|ctx| {
            let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: desc.label,
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(desc.source)),
            });
            let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: desc.label,
                layout: None,
                module: &module,
                entry_point: desc.entry_point,
            });
            let id = self.gen_id();
            self.resources.write().unwrap().compute_pipelines.insert(id, pipeline);
            id
        })
    }

    fn create_bind_group(&self, pipeline: PipelineId, group_index: u32, bindings: &[Binding]) -> Result<BindGroupId, BackendError> {
        self.with_context(|ctx| -> Result<BindGroupId, BackendError> {
            let res = self.resources.read().unwrap();
            let pipe = res.pipelines.get(&pipeline).ok_or(BackendError::Validation)?;
            let layout = pipe.get_bind_group_layout(group_index);

            let mut entries = Vec::new();
            for b in bindings {
                let resource = match b.resource {
                    BindingResource::Buffer(id) => wgpu::BindingResource::Buffer(
                        res.buffers.get(&id).ok_or(BackendError::Validation)?.as_entire_buffer_binding()
                    ),
                    BindingResource::Texture(id) => wgpu::BindingResource::TextureView(
                        res.texture_views.get(&id).ok_or(BackendError::Validation)?
                    ),
                    BindingResource::Sampler(id) => wgpu::BindingResource::Sampler(
                        res.samplers.get(&id).ok_or(BackendError::Validation)?
                    ),
                };
                entries.push(wgpu::BindGroupEntry { binding: b.binding, resource });
            }

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            });
            
            let id: BindGroupId = self.gen_id();
            drop(res);
            self.resources.write().unwrap().bind_groups.insert(id, bg);
            Ok(id)
        }).and_then(|res| res)
    }

    fn create_command_encoder(&self) -> Result<Box<dyn CommandEncoder>, BackendError> {
        self.with_context(|ctx| {
            let encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            Box::new(WgpuEncoder { encoder, resources: self.resources.clone() }) as Box<dyn CommandEncoder>
        })
    }

    fn submit(&self, encoder: Box<dyn CommandEncoder>) -> Result<(), BackendError> {
        self.with_context(|ctx| {
            // Unsafe downcast simulation to retrieve internal WGPU encoder
            // This is necessary because the trait object hides the concrete type
            let raw_encoder = unsafe {
                let ptr = Box::into_raw(encoder);
                *Box::from_raw(ptr as *mut WgpuEncoder)
            };
            let buf = raw_encoder.encoder.finish();
            ctx.queue.submit(std::iter::once(buf));
            
            if let Some(surf) = &ctx.surface {
                 // Simplified frame presentation logic
                 if let Ok(frame) = surf.get_current_texture() {
                     frame.present();
                 }
            }
            ()
        })
    }
}

// === ENCODER IMPLEMENTATION ===

struct WgpuEncoder {
    encoder: wgpu::CommandEncoder,
    resources: Arc<RwLock<Resources>>,
}

impl CommandEncoder for WgpuEncoder {
    fn begin_render_pass(&mut self, _desc: RenderPassDesc) -> Box<dyn RenderPass> {
        crate::agnesslog_unsupported!("RenderPass lifetime management in Rust traits is complex check lib code");
    }
    
    fn begin_compute_pass(&mut self) -> Box<dyn ComputePass> {
        crate::agnesslog_unsupported!("ComputePass impl");
    }

    fn copy_texture_to_texture(&mut self, src: TextureId, dst: TextureId, width: u32, height: u32, depth: u32) {
        let res = self.resources.read().unwrap();
        if let (Some(s), Some(d)) = (res.textures.get(&src), res.textures.get(&dst)) {
            self.encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture { texture: s, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::ImageCopyTexture { texture: d, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width, height, depth_or_array_layers: depth }
            );
        }
    }
}

// === MAPPERS ===

fn map_buffer_usage(u: BufferUsage) -> wgpu::BufferUsages {
    let mut out = wgpu::BufferUsages::COPY_DST;
    match u {
        BufferUsage::Vertex => out |= wgpu::BufferUsages::VERTEX,
        BufferUsage::Index => out |= wgpu::BufferUsages::INDEX,
        BufferUsage::Uniform => out |= wgpu::BufferUsages::UNIFORM,
        BufferUsage::Storage => out |= wgpu::BufferUsages::STORAGE,
        BufferUsage::Indirect => out |= wgpu::BufferUsages::INDIRECT,
        BufferUsage::Transfer => out |= wgpu::BufferUsages::COPY_SRC,
    }
    out
}

fn map_texture_format(f: TextureFormat) -> wgpu::TextureFormat {
    match f {
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
        TextureFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
        TextureFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
    }
}

fn map_address_mode(m: AddressMode) -> wgpu::AddressMode {
    match m {
        AddressMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        AddressMode::Repeat => wgpu::AddressMode::Repeat,
        AddressMode::MirrorRepeat => wgpu::AddressMode::MirrorRepeat,
    }
}

fn map_filter_mode(m: FilterMode) -> wgpu::FilterMode {
    match m {
        FilterMode::Nearest => wgpu::FilterMode::Nearest,
        FilterMode::Linear => wgpu::FilterMode::Linear,
    }
}

fn map_vertex_format(f: VertexFormat) -> wgpu::VertexFormat {
    match f {
        VertexFormat::FloatTwo => wgpu::VertexFormat::Float32x2,
        VertexFormat::FloatThree => wgpu::VertexFormat::Float32x3,
        VertexFormat::FloatFour => wgpu::VertexFormat::Float32x4,
        VertexFormat::UByteFour => wgpu::VertexFormat::Uint8x4,
        VertexFormat::UByteFourUint => wgpu::VertexFormat::Uint8x4,
        VertexFormat::ShortTwo => wgpu::VertexFormat::Sint16x2,
        VertexFormat::ShortFour => wgpu::VertexFormat::Sint16x4,
        VertexFormat::UShortTwo => wgpu::VertexFormat::Uint16x2,
        VertexFormat::UShortFour => wgpu::VertexFormat::Uint16x4,
        VertexFormat::ShortTwoNorm => wgpu::VertexFormat::Snorm16x2,
        VertexFormat::ShortFourNorm => wgpu::VertexFormat::Snorm16x4,
        VertexFormat::UShortTwoNorm => wgpu::VertexFormat::Unorm16x2,
        VertexFormat::UShortFourNorm => wgpu::VertexFormat::Unorm16x4,
        VertexFormat::Uint => wgpu::VertexFormat::Uint32,
        VertexFormat::UintTwo => wgpu::VertexFormat::Uint32x2,
        VertexFormat::UintThree => wgpu::VertexFormat::Uint32x3,
        VertexFormat::UintFour => wgpu::VertexFormat::Uint32x4,
    }
}

fn map_topology(t: PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match t {
        PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
    }
}

fn map_cull(c: CullMode) -> wgpu::Face {
    match c {
        CullMode::Front => wgpu::Face::Front,
        CullMode::Back => wgpu::Face::Back,
    }
}

fn map_compare(c: CompareFunction) -> wgpu::CompareFunction {
    match c {
        CompareFunction::Less => wgpu::CompareFunction::Less,
        CompareFunction::LessEqual => wgpu::CompareFunction::LessEqual,
        CompareFunction::Greater => wgpu::CompareFunction::Greater,
        CompareFunction::GreaterEqual => wgpu::CompareFunction::GreaterEqual,
        CompareFunction::Equal => wgpu::CompareFunction::Equal,
        CompareFunction::NotEqual => wgpu::CompareFunction::NotEqual,
        CompareFunction::Always => wgpu::CompareFunction::Always,
        CompareFunction::Never => wgpu::CompareFunction::Never,
    }
}

fn map_blend(b: BlendState) -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: map_blend_factor(b.color.src_factor),
            dst_factor: map_blend_factor(b.color.dst_factor),
            operation: map_blend_op(b.color.operation),
        },
        alpha: wgpu::BlendComponent {
            src_factor: map_blend_factor(b.alpha.src_factor),
            dst_factor: map_blend_factor(b.alpha.dst_factor),
            operation: map_blend_op(b.alpha.operation),
        },
    }
}

fn map_blend_factor(f: BlendFactor) -> wgpu::BlendFactor {
    match f {
        BlendFactor::Zero => wgpu::BlendFactor::Zero,
        BlendFactor::One => wgpu::BlendFactor::One,
        BlendFactor::SrcAlpha => wgpu::BlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => wgpu::BlendFactor::OneMinusSrcAlpha,
    }
}

fn map_blend_op(o: BlendOperation) -> wgpu::BlendOperation {
    match o {
        BlendOperation::Add => wgpu::BlendOperation::Add,
        BlendOperation::Subtract => wgpu::BlendOperation::Subtract,
    }
}