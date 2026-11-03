use std::path::PathBuf;

use anyhow::{Context, Result, bail};

pub enum Mode {
    Windowed,
    Probe,
    Screenshot(PathBuf),
}

pub struct Options {
    pub mode: Mode,
    pub view_radius: u32,
    pub vsync: bool,
}

const USAGE: &str = "usage: vx [--probe | --screenshot <file.png>] [--radius <chunks>] [--vsync]";

impl Options {
    pub fn from_args(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut options = Self {
            mode: Mode::Windowed,
            view_radius: 12,
            vsync: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--probe" => options.mode = Mode::Probe,
                "--screenshot" => {
                    options.mode = Mode::Screenshot(args.next().context(USAGE)?.into());
                }
                "--radius" => {
                    options.view_radius = args.next().context(USAGE)?.parse().context(USAGE)?;
                }
                "--vsync" => options.vsync = true,
                _ => bail!("unknown argument {arg:?}\n{USAGE}"),
            }
        }
        Ok(options)
    }
}
