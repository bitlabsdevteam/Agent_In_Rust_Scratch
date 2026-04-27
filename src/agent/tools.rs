use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Result, bail};
use chrono::Utc;

pub struct ToolContext {
    pub workspace_root: String,
}

#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn privileged(&self) -> bool {
        false
    }
    fn run(&self, args: &str, ctx: &ToolContext) -> Result<String>;
}

pub struct ToolRegistry {
    tools: HashMap<&'static str, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn default() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        registry.register(Box::new(EchoTool));
        registry.register(Box::new(TimeTool));
        registry.register(Box::new(ReadFileTool));
        registry
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name(), tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(Box::as_ref)
    }

    pub fn list(&self) -> Vec<(&'static str, &'static str, bool)> {
        let mut items: Vec<_> = self
            .tools
            .values()
            .map(|tool| (tool.name(), tool.description(), tool.privileged()))
            .collect();
        items.sort_by(|a, b| a.0.cmp(b.0));
        items
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut items: Vec<_> = self
            .tools
            .values()
            .map(|tool| ToolSpec {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
            })
            .collect();
        items.sort_by(|a, b| a.name.cmp(&b.name));
        items
    }
}

struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn description(&self) -> &'static str {
        "Returns the same text."
    }

    fn run(&self, args: &str, _ctx: &ToolContext) -> Result<String> {
        Ok(args.to_string())
    }
}

struct TimeTool;

impl Tool for TimeTool {
    fn name(&self) -> &'static str {
        "time"
    }

    fn description(&self) -> &'static str {
        "Returns current UTC timestamp."
    }

    fn run(&self, _args: &str, _ctx: &ToolContext) -> Result<String> {
        Ok(Utc::now().to_rfc3339())
    }
}

struct ReadFileTool;

impl Tool for ReadFileTool {
    fn name(&self) -> &'static str {
        "read_file"
    }

    fn description(&self) -> &'static str {
        "Reads a UTF-8 text file under workspace root."
    }

    fn privileged(&self) -> bool {
        true
    }

    fn run(&self, args: &str, ctx: &ToolContext) -> Result<String> {
        let rel = args.trim();
        if rel.is_empty() {
            bail!("read_file requires a relative path argument");
        }

        let candidate = Path::new(&ctx.workspace_root).join(rel);
        let canon_root = fs::canonicalize(&ctx.workspace_root)?;
        let canon_candidate = fs::canonicalize(&candidate)?;

        if !canon_candidate.starts_with(&canon_root) {
            bail!("path escapes workspace root");
        }

        let content = fs::read_to_string(canon_candidate)?;
        Ok(content)
    }
}
