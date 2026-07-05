pub mod game;
pub mod engine;

use engine::config::EngineConfig;
use engine::core::app::App;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu_hal::vulkan::instance", log::LevelFilter::Off)
        .filter_module("wgpu_core", log::LevelFilter::Error)
        .init();

    let config = EngineConfig::default();
    App::run(config);
}
