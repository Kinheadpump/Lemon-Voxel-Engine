use winit::keyboard::KeyCode;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MoveCommand {
    Forward,
    Backward,
    StrafeLeft,
    StrafeRight,
    Ascend,
    Descend,
}

#[derive(Default)]
pub struct InputState {
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
    sprinting: bool,
    mouse_delta: (f32, f32),
    hud_toggle_requested: bool,
}

impl InputState {
    pub fn handle_key(&mut self, key: KeyCode, pressed: bool) {
        match key {
            KeyCode::KeyW => self.move_forward = pressed,
            KeyCode::KeyS => self.move_backward = pressed,
            KeyCode::KeyA => self.move_left = pressed,
            KeyCode::KeyD => self.move_right = pressed,
            KeyCode::Space => self.move_up = pressed,
            KeyCode::ControlLeft | KeyCode::ControlRight => self.move_down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.sprinting = pressed,
            KeyCode::F3 if pressed => self.hud_toggle_requested = true,
            _ => {}
        }
    }

    pub fn is_sprinting(&self) -> bool {
        self.sprinting
    }

    pub fn take_hud_toggle_requested(&mut self) -> bool {
        std::mem::take(&mut self.hud_toggle_requested)
    }

    pub fn handle_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    pub fn take_mouse_delta(&mut self) -> (f32, f32) {
        std::mem::take(&mut self.mouse_delta)
    }

    pub fn active_commands(&self) -> Vec<MoveCommand> {
        let mut commands = Vec::with_capacity(6);
        if self.move_forward {
            commands.push(MoveCommand::Forward);
        }
        if self.move_backward {
            commands.push(MoveCommand::Backward);
        }
        if self.move_left {
            commands.push(MoveCommand::StrafeLeft);
        }
        if self.move_right {
            commands.push(MoveCommand::StrafeRight);
        }
        if self.move_up {
            commands.push(MoveCommand::Ascend);
        }
        if self.move_down {
            commands.push(MoveCommand::Descend);
        }
        commands
    }
}
