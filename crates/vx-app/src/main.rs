mod headless;
mod options;
mod region;
mod windowed;

use anyhow::Result;
use options::{Mode, Options};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let options = Options::from_args(std::env::args().skip(1))?;
    match &options.mode {
        Mode::Probe => headless::probe(),
        Mode::Screenshot(path) => headless::screenshot(&options, path),
        Mode::Windowed => windowed::run(options),
    }
}
