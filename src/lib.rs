pub mod error;
pub mod debug;
pub mod backend;

use std::num::NonZeroU64;
use std::ops::Range;

use raw_window_handle::HasWindowHandle;
use raw_window_handle::HasDisplayHandle;

use crate::error::BackendError;
use crate::backend::BackendSelector;

/// Unique resource identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(NonZeroU64);

impl ResourceId {
    /// Creates a new identifier from a numeric value
    pub fn new(id: u64) -> Option<Self> {
        NonZeroU64::new(id).map(ResourceId)
    }
}

// Handle definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub ResourceId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(pub ResourceId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerId(pub ResourceId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineId(pub ResourceId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindGroupId(pub ResourceId);

macro_rules! impl_handle_from {
    ($name:ident) => {
        impl From<ResourceId> for $name {
            fn from(id: ResourceId) -> Self { Self(id) }
        }
    };
}

impl_handle_from!(BufferId);
impl_handle_from!(TextureId);
impl_handle_from!(SamplerId);
impl_handle_from!(PipelineId);
impl_handle_from!(BindGroupId);

/// Buffer usage defines allowed operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    Vertex, // Geometry data
    Index, // Index data
    Uniform, // Constants
    Storage, // Read Write arrays
    Indirect, // GPU commands
    Transfer, // Copy operations
}

/// Texture type defines dimensions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureType {
    TwoDimensions,
    ThreeDimensions,
    Cube,
    Array,
}

/// Pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8Unorm,
    Bgra8Unorm,
    Rgba16Float,
    Depth32Float,
    R8Unorm,
}

/// Texture addressing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
}

/// Texture filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Nearest,
    Linear,
}

/// Buffer creation descriptor
#[derive(Debug, Clone)]
pub struct BufferDesc {
    pub size: u64,
    pub usage: BufferUsage,
    pub label: Option<String>,
}

/// Texture creation descriptor
#[derive(Debug, Clone)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub ty: TextureType,
    pub format: TextureFormat,
    pub render_target: bool,
    pub label: Option<String>,
}

/// Sampler descriptor
#[derive(Debug, Clone)]
pub struct SamplerDesc {
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub address_w: AddressMode,
    pub mag_filter: FilterMode,
    pub min_filter: FilterMode,
    pub anisotropy_clamp: u16,
}

/// Graphics pipeline descriptor
#[derive(Debug, Clone)]
pub struct RenderPipelineDesc<'a> {
    pub label: Option<&'a str>,
    pub vs_source: &'a str,
    pub fs_source: &'a str,
    /// List of buffer layouts
    /// [*] Use multiple layouts to split Vertex and Instance data
    /// to avoid vertex size limits and optimize bandwidth
    pub vertex_layouts: &'a [VertexBufferLayout<'a>],
    pub primitive: PrimitiveState,
    pub depth_stencil: Option<DepthStencilState>,
    pub blend: Option<BlendState>,
}

/// Layout of a single vertex buffer
#[derive(Debug, Clone)]
pub struct VertexBufferLayout<'a> {
    /// Stride in bytes between elements
    pub array_stride: u64,
    /// How to move through the buffer
    pub step_mode: VertexStepMode,
    /// List of attributes within this buffer
    pub attributes: &'a [VertexAttribute],
}

/// Step mode for vertex buffer
#[derive(Debug, Clone, Copy)]
pub enum VertexStepMode {
    Vertex, // Advance per vertex
    Instance, // Advance per instance
}

/// Description of a single vertex attribute
#[derive(Debug, Clone, Copy)]
pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub location: u32,
}

/// Data format for vertex attributes
/// [*] Expanded to support packed types for optimization
#[derive(Debug, Clone, Copy)]
pub enum VertexFormat {
    FloatTwo, // Two floats
    FloatThree, // Three floats
    FloatFour, // Four floats
    
    UByteFour, // Four bytes normalized
    UByteFourUint, // Four bytes unsigned integer
    
    ShortTwo, // Two signed shorts
    ShortFour, // Four signed shorts
    
    UShortTwo, // Two unsigned shorts
    UShortFour, // Four unsigned shorts
    
    ShortTwoNorm, // Two signed normalized shorts
    ShortFourNorm, // Four signed normalized shorts
    
    UShortTwoNorm, // Two unsigned normalized shorts
    UShortFourNorm, // Four unsigned normalized shorts

    Uint, // Single unsigned int
    UintTwo, // Two unsigned ints
    UintThree, // Three unsigned ints
    UintFour, // Four unsigned ints
}

/// Primitive topology and culling
#[derive(Debug, Clone, Copy)]
pub struct PrimitiveState {
    pub topology: PrimitiveTopology,
    pub cull_mode: Option<CullMode>,
    pub front_face: FrontFace,
}

#[derive(Debug, Clone, Copy)]
pub enum PrimitiveTopology {
    TriangleList,
    LineList,
    PointList,
}

#[derive(Debug, Clone, Copy)]
pub enum CullMode {
    Front,
    Back,
}

#[derive(Debug, Clone, Copy)]
pub enum FrontFace {
    Ccw,
    Cw,
}

/// Depth Stencil state
#[derive(Debug, Clone, Copy)]
pub struct DepthStencilState {
    pub format: TextureFormat,
    pub depth_write: bool,
    pub depth_compare: CompareFunction,
}

#[derive(Debug, Clone, Copy)]
pub enum CompareFunction {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Always,
    Never,
}

/// Blending state
#[derive(Debug, Clone, Copy)]
pub struct BlendState {
    pub alpha: BlendComponent,
    pub color: BlendComponent,
}

#[derive(Debug, Clone, Copy)]
pub struct BlendComponent {
    pub src_factor: BlendFactor,
    pub dst_factor: BlendFactor,
    pub operation: BlendOperation,
}

#[derive(Debug, Clone, Copy)]
pub enum BlendFactor {
    Zero,
    One,
    SrcAlpha,
    OneMinusSrcAlpha,
}

#[derive(Debug, Clone, Copy)]
pub enum BlendOperation {
    Add,
    Subtract,
}

/// Compute pipeline descriptor
#[derive(Debug, Clone)]
pub struct ComputePipelineDesc<'a> {
    pub label: Option<&'a str>,
    pub source: &'a str,
    pub entry_point: &'a str,
}

/// Resource binding
#[derive(Debug, Clone)]
pub struct Binding {
    pub binding: u32,
    pub resource: BindingResource,
}

#[derive(Debug, Clone)]
pub enum BindingResource {
    Buffer(BufferId),
    Texture(TextureId),
    Sampler(SamplerId),
}

/// Render pass attachments
#[derive(Debug, Clone)]
pub struct RenderPassDesc<'a> {
    pub color_attachments: &'a [ColorAttachment],
    pub depth_attachment: Option<DepthAttachment>,
}

#[derive(Debug, Clone)]
pub struct ColorAttachment {
    pub target: RenderTarget,
    pub load_op: LoadOp,
    pub clear_value: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct DepthAttachment {
    pub target: TextureId,
    pub load_op: LoadOp,
    pub clear_value: f32,
    pub store: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum RenderTarget {
    Screen,
    Texture(TextureId),
}

#[derive(Debug, Clone, Copy)]
pub enum LoadOp {
    Clear,
    Load,
}

/// Unified entry point to initialize the graphics engine
/// [*] Returns a boxed trait object to allow runtime backend selection
pub fn new<W>(
    selector: BackendSelector,
    window: Option<&W>,
) -> Result<Box<dyn AgnessBackend>, BackendError>
where
    W: HasWindowHandle + HasDisplayHandle,
{
    match selector {
        BackendSelector::Wgpu => {
            // Check feature presence inside the match arm
            
            // Branch one Feature is enabled
            #[cfg(feature = "wgpu")]
            {
                let instance = backend::wgpu::WgpuBackend::new(window)?;
                Ok(Box::new(instance))
            }

            // Branch two Feature is disabled
            // [*] We panic here because caller requested a backend that was not compiled in
            #[cfg(not(feature = "wgpu"))]
            {
                crate::agnesslog_unsupported!("WGPU feature is disabled in Cargo configuration");
            }
        }
    }
}

/// Main backend trait
pub trait AgnessBackend: Send + Sync {
    fn new<W>(window: Option<&W>) -> Result<Self, BackendError>
    where
        W: HasWindowHandle + HasDisplayHandle,
        Self: Sized;

    fn create_buffer(&self, desc: BufferDesc) -> Result<BufferId, BackendError>;
    fn write_buffer(&self, buffer: BufferId, offset: u64, data: &[u8]) -> Result<(), BackendError>;

    fn create_texture(&self, desc: TextureDesc) -> Result<TextureId, BackendError>;
    fn write_texture(&self, texture: TextureId, data: &[u8], bytes_per_row: u32) -> Result<(), BackendError>;

    fn create_sampler(&self, desc: SamplerDesc) -> Result<SamplerId, BackendError>;

    fn create_render_pipeline(&self, desc: RenderPipelineDesc) -> Result<PipelineId, BackendError>;
    fn create_compute_pipeline(&self, desc: ComputePipelineDesc) -> Result<PipelineId, BackendError>;

    fn create_bind_group(&self, pipeline: PipelineId, group_index: u32, bindings: &[Binding]) -> Result<BindGroupId, BackendError>;

    fn create_command_encoder(&self) -> Result<Box<dyn CommandEncoder>, BackendError>;
    fn submit(&self, encoder: Box<dyn CommandEncoder>) -> Result<(), BackendError>;
}

pub trait CommandEncoder {
    fn begin_render_pass(&mut self, desc: RenderPassDesc) -> Box<dyn RenderPass>;
    fn begin_compute_pass(&mut self) -> Box<dyn ComputePass>;
    fn copy_texture_to_texture(&mut self, src: TextureId, dst: TextureId, width: u32, height: u32, depth: u32);
}

pub trait RenderPass {
    fn set_pipeline(&mut self, pipeline: PipelineId);
    fn set_bind_group(&mut self, index: u32, group: BindGroupId);
    
    /// Sets a vertex buffer for a specific slot
    /// [*] Use slot zero for geometry and slot one for instance data
    fn set_vertex_buffer(&mut self, slot: u32, buffer: BufferId);
    fn set_index_buffer(&mut self, buffer: BufferId);
    
    fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32);
    fn set_scissor(&mut self, x: u32, y: u32, w: u32, h: u32);
    
    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>);
    fn draw_indexed(&mut self, indices: Range<u32>, instances: Range<u32>);
    fn draw_indirect(&mut self, buffer: BufferId, offset: u64);
    fn draw_indexed_indirect(&mut self, buffer: BufferId, offset: u64);
}

pub trait ComputePass {
    fn set_pipeline(&mut self, pipeline: PipelineId);
    fn set_bind_group(&mut self, index: u32, group: BindGroupId);
    fn dispatch(&mut self, x: u32, y: u32, z: u32);
    fn dispatch_indirect(&mut self, buffer: BufferId, offset: u64);
}