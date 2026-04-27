mod agent;

use std::env;

use anyhow::Result;
use dotenvy::dotenv;

use agent::harness::Harness;
use agent::model::AnyModel;
use agent::permissions::PermissionPolicy;
use agent::session::SessionStore;
use agent::tools::{ToolContext, ToolRegistry};

fn main() -> Result<()> {
    dotenv().ok();

    let cwd = env::current_dir()?;
    let workspace_root = cwd.to_string_lossy().to_string();
    let session_file = cwd.join(".agent").join("session.jsonl");
    let model = AnyModel::from_env()?;

    let mut harness = Harness::new(
        model,
        ToolRegistry::default(),
        PermissionPolicy::default(),
        SessionStore::new(session_file),
        ToolContext { workspace_root },
        60,
    )?;

    harness.run()
}
