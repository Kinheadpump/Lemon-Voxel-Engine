mod camera;
mod gpu;
mod offscreen;
mod terrain;

pub use camera::{CameraUniform, FlyCamera};
pub use gpu::{GpuContext, RenderTier};
pub use offscreen::{OffscreenTarget, begin_terrain_pass, create_depth_view};
pub use terrain::{ChunkMeshRef, DEPTH_FORMAT, TerrainRenderer};

pub const TERRAIN_SHADER: &str = include_str!("shaders/terrain.wgsl");
