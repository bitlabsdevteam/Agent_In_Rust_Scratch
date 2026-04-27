use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HarnessConfig {
    pub compact_threshold_messages: usize,
    pub max_inbound_events_per_cycle: usize,
    pub max_outbound_events_per_cycle: usize,
    pub heartbeat_interval_secs: u64,
    pub scheduler_enabled: bool,
    pub max_scheduler_pending_inbound: usize,
    pub websocket_enabled: bool,
    pub websocket_bind_addr: String,
    pub websocket_auth_required: bool,
    pub websocket_auth_token: String,
    pub websocket_auth_previous_token: String,
    pub websocket_auth_rotation_grace_secs: u64,
    pub websocket_max_clients: usize,
    pub websocket_idle_timeout_secs: u64,
    pub websocket_ping_interval_secs: u64,
    pub telegram_enabled: bool,
    pub telegram_bot_token: String,
    pub telegram_api_base_url: String,
    pub telegram_poll_interval_secs: u64,
    pub telegram_poll_timeout_secs: u64,
    pub telegram_allowed_chat_ids: Vec<i64>,
    pub memory_retention_ttl_days: i64,
    pub memory_retention_max_records: usize,
    pub memory_search_mode: String,
    pub memory_hybrid_lexical_weight: f64,
    pub memory_hybrid_semantic_weight: f64,
    pub memory_embedding_provider: String,
    pub memory_embedding_model: String,
    pub memory_embedding_base_url: String,
    pub memory_embedding_api_key_env: String,
    pub memory_embedding_dimensions: usize,
    pub memory_ann_enabled: bool,
    pub memory_ann_scan_threshold: usize,
    pub memory_ann_probe_count: usize,
    pub memory_ann_candidate_cap: usize,
    pub memory_ann_backfill_batch_size: usize,
    pub memory_ann_maintenance_interval_ticks: u64,
    pub memory_status_snapshot_interval_ticks: u64,
    pub skill_dir: String,
    pub policy_file: String,
    pub artifacts_dir: String,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            compact_threshold_messages: 60,
            max_inbound_events_per_cycle: 64,
            max_outbound_events_per_cycle: 64,
            heartbeat_interval_secs: 300,
            scheduler_enabled: true,
            max_scheduler_pending_inbound: 64,
            websocket_enabled: true,
            websocket_bind_addr: "127.0.0.1:9001".to_string(),
            websocket_auth_required: false,
            websocket_auth_token: "".to_string(),
            websocket_auth_previous_token: "".to_string(),
            websocket_auth_rotation_grace_secs: 300,
            websocket_max_clients: 128,
            websocket_idle_timeout_secs: 300,
            websocket_ping_interval_secs: 30,
            telegram_enabled: false,
            telegram_bot_token: "".to_string(),
            telegram_api_base_url: "https://api.telegram.org".to_string(),
            telegram_poll_interval_secs: 2,
            telegram_poll_timeout_secs: 0,
            telegram_allowed_chat_ids: Vec::new(),
            memory_retention_ttl_days: 30,
            memory_retention_max_records: 2000,
            memory_search_mode: "hybrid".to_string(),
            memory_hybrid_lexical_weight: 0.6,
            memory_hybrid_semantic_weight: 0.4,
            memory_embedding_provider: "local".to_string(),
            memory_embedding_model: "text-embedding-3-small".to_string(),
            memory_embedding_base_url: "https://api.openai.com/v1".to_string(),
            memory_embedding_api_key_env: "OPENAI_API_KEY".to_string(),
            memory_embedding_dimensions: 128,
            memory_ann_enabled: true,
            memory_ann_scan_threshold: 200,
            memory_ann_probe_count: 6,
            memory_ann_candidate_cap: 256,
            memory_ann_backfill_batch_size: 32,
            memory_ann_maintenance_interval_ticks: 20,
            memory_status_snapshot_interval_ticks: 0,
            skill_dir: ".agent/skills".to_string(),
            policy_file: ".agent/policy.json".to_string(),
            artifacts_dir: ".agent/artifacts".to_string(),
        }
    }
}

pub struct ConfigManager {
    path: PathBuf,
    last_modified: Option<SystemTime>,
}

impl ConfigManager {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            last_modified: None,
        }
    }

    pub fn load(&mut self) -> Result<HarnessConfig> {
        let cfg = load_config_or_default(&self.path)?;
        self.last_modified = modified_time(&self.path)?;
        Ok(cfg)
    }

    pub fn reload_if_changed(&mut self) -> Result<Option<HarnessConfig>> {
        let current = modified_time(&self.path)?;
        if current != self.last_modified {
            let cfg = load_config_or_default(&self.path)?;
            self.last_modified = current;
            return Ok(Some(cfg));
        }
        Ok(None)
    }
}

fn modified_time(path: &Path) -> Result<Option<SystemTime>> {
    if !path.exists() {
        return Ok(None);
    }
    let modified = fs::metadata(path)
        .with_context(|| format!("failed to read metadata: {}", path.display()))?
        .modified()
        .with_context(|| format!("failed to read modified time: {}", path.display()))?;
    Ok(Some(modified))
}

pub fn load_config_or_default(path: &Path) -> Result<HarnessConfig> {
    if !path.exists() {
        return Ok(HarnessConfig::default());
    }
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;
    let cfg: HarnessConfig = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse config json: {}", path.display()))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{ConfigManager, HarnessConfig, load_config_or_default};

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
    fn missing_config_uses_defaults() {
        let dir = temp_test_dir("config_defaults");
        let path = dir.join("missing.json");

        let cfg = load_config_or_default(&path).expect("load should succeed");
        assert_eq!(cfg.compact_threshold_messages, 60);
        assert_eq!(cfg.skill_dir, ".agent/skills");
        assert_eq!(cfg.websocket_max_clients, 128);

        fs::remove_dir_all(dir).expect("failed to cleanup temp dir");
    }

    #[test]
    fn reload_if_changed_detects_file_updates() {
        let dir = temp_test_dir("config_reload");
        let path = dir.join("agent.json");
        let cfg = HarnessConfig {
            max_inbound_events_per_cycle: 1,
            ..HarnessConfig::default()
        };
        fs::write(
            &path,
            serde_json::to_string_pretty(&cfg).expect("config serialization should succeed"),
        )
        .expect("config write should succeed");

        let mut manager = ConfigManager::new(&path);
        let first = manager.load().expect("initial load should succeed");
        assert_eq!(first.max_inbound_events_per_cycle, 1);

        let cfg2 = HarnessConfig {
            max_inbound_events_per_cycle: 5,
            ..HarnessConfig::default()
        };
        std::thread::sleep(std::time::Duration::from_millis(5));
        fs::write(
            &path,
            serde_json::to_string_pretty(&cfg2).expect("config serialization should succeed"),
        )
        .expect("config rewrite should succeed");

        let updated = manager
            .reload_if_changed()
            .expect("reload should succeed")
            .expect("config should be reloaded");
        assert_eq!(updated.max_inbound_events_per_cycle, 5);

        fs::remove_dir_all(dir).expect("failed to cleanup temp dir");
    }

    #[test]
    fn partial_config_json_uses_defaults_for_missing_fields() {
        let dir = temp_test_dir("config_partial");
        let path = dir.join("agent.json");
        fs::write(&path, r#"{"max_inbound_events_per_cycle": 3}"#)
            .expect("config write should succeed");

        let cfg = load_config_or_default(&path).expect("load should succeed");
        assert_eq!(cfg.max_inbound_events_per_cycle, 3);
        assert_eq!(cfg.memory_retention_ttl_days, 30);
        assert_eq!(cfg.memory_retention_max_records, 2000);
        assert_eq!(cfg.websocket_idle_timeout_secs, 300);
        assert_eq!(cfg.websocket_ping_interval_secs, 30);
        assert!(!cfg.telegram_enabled);
        assert_eq!(cfg.telegram_api_base_url, "https://api.telegram.org");
        assert_eq!(cfg.telegram_poll_interval_secs, 2);
        assert_eq!(cfg.telegram_poll_timeout_secs, 0);
        assert!(cfg.telegram_allowed_chat_ids.is_empty());
        assert_eq!(cfg.memory_search_mode, "hybrid");
        assert!((cfg.memory_hybrid_lexical_weight - 0.6).abs() < f64::EPSILON);
        assert!((cfg.memory_hybrid_semantic_weight - 0.4).abs() < f64::EPSILON);
        assert_eq!(cfg.memory_embedding_provider, "local");
        assert_eq!(cfg.memory_embedding_dimensions, 128);
        assert!(cfg.memory_ann_enabled);
        assert_eq!(cfg.memory_ann_scan_threshold, 200);
        assert_eq!(cfg.memory_ann_backfill_batch_size, 32);
        assert_eq!(cfg.memory_ann_maintenance_interval_ticks, 20);
        assert_eq!(cfg.memory_status_snapshot_interval_ticks, 0);

        fs::remove_dir_all(dir).expect("failed to cleanup temp dir");
    }
}
