use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct PermissionPolicy {
    pub ask_before_privileged_tools: bool,
    pub privileged_allowlist: HashSet<String>,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            ask_before_privileged_tools: true,
            privileged_allowlist: HashSet::new(),
        }
    }
}

impl PermissionPolicy {
    pub fn should_require_approval(&self, tool_name: &str, privileged: bool) -> bool {
        if !privileged || !self.ask_before_privileged_tools {
            return false;
        }
        !self.privileged_allowlist.contains(tool_name)
    }

    pub fn approve_tool(&mut self, tool_name: impl Into<String>) {
        self.privileged_allowlist.insert(tool_name.into());
    }
}
