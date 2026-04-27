#[derive(Debug, Clone)]
pub struct PromptLayers {
    pub identity: String,
    pub channel: String,
    pub skills: String,
    pub memory: String,
}

pub fn assemble_prompt(layers: &PromptLayers) -> String {
    format!(
        "# Identity\n{}\n\n# Channel\n{}\n\n# Skills\n{}\n\n# Memory\n{}",
        layers.identity, layers.channel, layers.skills, layers.memory
    )
}

#[cfg(test)]
mod tests {
    use super::{PromptLayers, assemble_prompt};

    #[test]
    fn assembled_prompt_contains_all_layers() {
        let out = assemble_prompt(&PromptLayers {
            identity: "agent".to_string(),
            channel: "cli".to_string(),
            skills: "none".to_string(),
            memory: "recent context".to_string(),
        });

        assert!(out.contains("# Identity"));
        assert!(out.contains("# Skills"));
        assert!(out.contains("recent context"));
    }
}
