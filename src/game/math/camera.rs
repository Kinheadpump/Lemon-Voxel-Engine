use glam::{Mat4, Vec3};

const PITCH_LIMIT_RADIANS: f32 = 1.55;

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_radians: f32,
    pub near: f32,
}

impl Camera {
    pub fn new(position: Vec3, yaw: f32, pitch: f32, fov_y_radians: f32) -> Self {
        Self { position, yaw, pitch, fov_y_radians, near: 0.1 }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    /// Vorwaerts-Vektor projiziert auf die XZ-Ebene, fuer Spectator-Bewegung unabhaengig vom Pitch.
    pub fn forward_flat(&self) -> Vec3 {
        let forward = self.forward();
        let flattened = Vec3::new(forward.x, 0.0, forward.z);
        if flattened.length_squared() > 1e-6 {
            flattened.normalize()
        } else {
            Vec3::ZERO
        }
    }

    pub fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    pub fn rotate(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += delta_yaw;
        self.pitch = (self.pitch + delta_pitch).clamp(-PITCH_LIMIT_RADIANS, PITCH_LIMIT_RADIANS);
    }

    pub fn view_matrix(&self) -> Mat4 {
        glam::camera::rh::view::look_to_mat4(self.position, self.forward(), Vec3::Y)
    }

    pub fn projection_matrix(&self, aspect: f32) -> Mat4 {
        glam::camera::rh::proj::directx::perspective_infinite_reverse(
            self.fov_y_radians,
            aspect,
            self.near,
        )
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection_matrix(aspect) * self.view_matrix()
    }
}
