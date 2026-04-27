use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Debug)]
pub struct SkillRegistry {
    pub skills: Vec<SkillManifest>,
}

impl SkillRegistry {
    pub fn load_from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Ok(Self { skills: Vec::new() });
        }

        let mut skills = Vec::new();
        let mut seen = HashSet::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }

            let text = fs::read_to_string(&path)?;
            let skill: SkillManifest = serde_json::from_str(&text)
                .map_err(|e| anyhow!("failed to parse skill {}: {e}", path.display()))?;

            if skill.name.trim().is_empty() {
                return Err(anyhow!("skill name is empty in {}", path.display()));
            }
            if !seen.insert(skill.name.clone()) {
                return Err(anyhow!("duplicate skill name: {}", skill.name));
            }

            skills.push(skill);
        }

        skills.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { skills })
    }

    pub fn summary_for_prompt(&self) -> String {
        if self.skills.is_empty() {
            return "No skills loaded.".to_string();
        }

        let mut lines = Vec::new();
        for skill in &self.skills {
            let caps = if skill.capabilities.is_empty() {
                "general".to_string()
            } else {
                skill.capabilities.join(", ")
            };
            lines.push(format!(
                "- {}: {} [{}]",
                skill.name, skill.description, caps
            ));
        }
        lines.join("\n")
    }

    pub fn names(&self) -> Vec<String> {
        self.skills.iter().map(|s| s.name.clone()).collect()
    }
}

pub fn default_skill_dir(workspace_root: &str, rel: &str) -> PathBuf {
    Path::new(workspace_root).join(rel)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::SkillRegistry;

    fn temp_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "agent_in_rust_scratch_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&dir).expect("failed to create temp test directory");
        dir
    }

    #[test]
    fn load_skills_reads_json_manifests() {
        let dir = temp_test_dir("skills_load");
        fs::write(
            dir.join("search.json"),
            r#"{"name":"search","description":"Search docs","capabilities":["lookup"]}"#,
        )
        .expect("write skill manifest should succeed");

        let registry = SkillRegistry::load_from_dir(&dir).expect("load skills should succeed");
        assert_eq!(registry.names(), vec!["search".to_string()]);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn duplicate_skill_name_rejected() {
        let dir = temp_test_dir("skills_dup");
        fs::write(
            dir.join("a.json"),
            r#"{"name":"dup","description":"A","capabilities":[]}"#,
        )
        .expect("write first manifest should succeed");
        fs::write(
            dir.join("b.json"),
            r#"{"name":"dup","description":"B","capabilities":[]}"#,
        )
        .expect("write second manifest should succeed");

        let err = SkillRegistry::load_from_dir(&dir).expect_err("duplicate names should fail");
        assert!(format!("{err:#}").contains("duplicate skill name"));

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }
}
