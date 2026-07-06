use voxel_engine::engine::config::{CONFIG_PATH, EngineConfig};
use voxel_engine::engine::core::app::App;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu_hal::vulkan::instance", log::LevelFilter::Off)
        .filter_module("wgpu_core", log::LevelFilter::Error)
        .init();

    let config = EngineConfig::load_or_create(CONFIG_PATH);
    App::run(config);
}
