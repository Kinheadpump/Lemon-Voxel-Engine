struct Camera {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> quads: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> chunk_origins: array<vec4<i32>>;

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) @interpolate(flat) block: u32,
    @location(2) @interpolate(flat) direction: u32,
}

const CORNERS_FRONT = array<vec2<u32>, 6>(
    vec2(0u, 0u), vec2(1u, 0u), vec2(1u, 1u), vec2(0u, 0u), vec2(1u, 1u), vec2(0u, 1u),
);
const CORNERS_FLIPPED = array<vec2<u32>, 6>(
    vec2(0u, 0u), vec2(1u, 1u), vec2(1u, 0u), vec2(0u, 0u), vec2(0u, 1u), vec2(1u, 1u),
);

// Bit layout mirrors vx_mesh::PackedQuad: lo = x6 y6 z6 (w-1)6 (h-1)6 dir[0..2]; hi = dir[2] block16.
fn depth_u_v_to_xyz(axis: u32, d: u32, u: u32, v: u32) -> vec3<u32> {
    switch axis {
        case 0u: { return vec3(d, u, v); }
        case 1u: { return vec3(u, d, v); }
        default: { return vec3(u, v, d); }
    }
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) chunk_slot: u32) -> VertexOut {
    let quad = quads[vertex_index / 6u];
    let lo = quad.x;
    let hi = quad.y;
    let origin = vec3(lo & 63u, (lo >> 6u) & 63u, (lo >> 12u) & 63u);
    let size = vec2(((lo >> 18u) & 63u) + 1u, ((lo >> 24u) & 63u) + 1u);
    let direction = (lo >> 30u) | ((hi & 1u) << 2u);
    let block = (hi >> 1u) & 0xFFFFu;

    let axis = direction >> 1u;
    let positive = (direction & 1u) == 0u;
    // Quad winding: u x v points along +axis for X and Z but along -axis for Y.
    let flip = positive == (axis == 1u);
    var corner = CORNERS_FRONT[vertex_index % 6u];
    if flip {
        corner = CORNERS_FLIPPED[vertex_index % 6u];
    }

    var depth_u_v: vec3<u32>;
    switch axis {
        case 0u: { depth_u_v = vec3(origin.x, origin.y, origin.z); }
        case 1u: { depth_u_v = vec3(origin.y, origin.x, origin.z); }
        default: { depth_u_v = vec3(origin.z, origin.x, origin.y); }
    }
    let local = depth_u_v_to_xyz(
        axis,
        depth_u_v.x + select(0u, 1u, positive),
        depth_u_v.y + corner.x * size.x,
        depth_u_v.z + corner.y * size.y,
    );
    let world = vec3<f32>(chunk_origins[chunk_slot].xyz) + vec3<f32>(local);

    var out: VertexOut;
    out.clip_position = camera.view_proj * vec4(world, 1.0);
    out.world_position = world;
    out.block = block;
    out.direction = direction;
    return out;
}

const BLOCK_COLORS = array<vec3<f32>, 6>(
    vec3(1.0, 0.0, 1.0),
    vec3(0.50, 0.50, 0.52),
    vec3(0.47, 0.33, 0.22),
    vec3(0.33, 0.62, 0.24),
    vec3(0.86, 0.80, 0.55),
    vec3(0.95, 0.96, 0.98),
);
const DIRECTION_SHADE = array<f32, 6>(0.80, 0.80, 1.00, 0.50, 0.65, 0.65);
const SKY_COLOR = vec3(0.34, 0.54, 0.83);
const FOG_DENSITY = 0.0012;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let albedo = pow(BLOCK_COLORS[min(in.block, 5u)], vec3(2.2));
    let base = albedo * DIRECTION_SHADE[in.direction];
    let distance = length(in.world_position - camera.position.xyz);
    let fog = 1.0 - exp(-distance * FOG_DENSITY);
    return vec4(mix(base, SKY_COLOR, fog), 1.0);
}
