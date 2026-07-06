use voxel_engine::engine::config::EngineConfig;
use voxel_engine::engine::core::app::App;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .filter_module("wgpu_hal::vulkan::instance", log::LevelFilter::Off)
        .filter_module("wgpu_core", log::LevelFilter::Error)
        .init();

    let config = EngineConfig::default();
    App::run(config);
}
