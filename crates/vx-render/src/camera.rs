use bytemuck::{Pod, Zeroable};
use glam::camera::rh::{proj::directx, view};
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: Mat4,
    pub position: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct FlyCamera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub vertical_fov: f32,
    pub near: f32,
}

impl FlyCamera {
    pub fn looking_at(position: Vec3, target: Vec3) -> Self {
        let forward = (target - position).normalize();
        Self {
            position,
            yaw: forward.z.atan2(forward.x),
            pitch: forward.y.asin(),
            vertical_fov: 70f32.to_radians(),
            near: 0.1,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
    }

    pub fn flat_right(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, self.yaw.cos())
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn uniform(&self, aspect: f32) -> CameraUniform {
        let view = view::look_to_mat4(self.position, self.forward(), Vec3::Y);
        let projection =
            directx::perspective_infinite_reverse(self.vertical_fov, aspect, self.near);
        CameraUniform {
            view_proj: projection * view,
            position: self.position.extend(1.0).to_array(),
        }
    }
}
