use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Effect {
    Allow,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ActionMatcher {
    #[default]
    Exact,
    Prefix,
    Wildcard,
    Regex,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuleScope {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    pub effect: Effect,
    pub action: String,
    #[serde(default)]
    pub action_matcher: ActionMatcher,
    #[serde(default)]
    pub scope: RuleScope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    #[serde(default = "default_allow")]
    pub default_effect: Effect,
    #[serde(default)]
    pub rules: Vec<PolicyRule>,
}

fn default_allow() -> Effect {
    Effect::Allow
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            default_effect: Effect::Allow,
            rules: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PolicyContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<String>,
}

impl PolicyContext {
    pub fn new(source: Option<&str>, channel: Option<&str>, agent: Option<&str>) -> Self {
        Self {
            source: source.map(ToString::to_string),
            channel: channel.map(ToString::to_string),
            agent: agent.map(ToString::to_string),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PolicyDecision {
    pub allowed: bool,
    pub reason: String,
}

pub struct PolicyEngine {
    config_path: PathBuf,
    audit_path: PathBuf,
    config: PolicyConfig,
}

impl PolicyEngine {
    pub fn new(config_path: impl AsRef<Path>, audit_path: impl AsRef<Path>) -> Result<Self> {
        let config_path = config_path.as_ref().to_path_buf();
        let audit_path = audit_path.as_ref().to_path_buf();
        let config = load_policy_or_default(&config_path)?;
        Ok(Self {
            config_path,
            audit_path,
            config,
        })
    }

    pub fn reload(&mut self) -> Result<()> {
        self.config = load_policy_or_default(&self.config_path)?;
        Ok(())
    }

    pub fn evaluate(&self, action: &str, context: &PolicyContext) -> PolicyDecision {
        for rule in &self.config.rules {
            if !match_action(rule, action) {
                continue;
            }
            if !match_scope(&rule.scope, context) {
                continue;
            }

            return match rule.effect {
                Effect::Allow => PolicyDecision {
                    allowed: true,
                    reason: format!(
                        "allow rule matched: {} ({:?})",
                        rule.action, rule.action_matcher
                    ),
                },
                Effect::Deny => PolicyDecision {
                    allowed: false,
                    reason: format!(
                        "deny rule matched: {} ({:?})",
                        rule.action, rule.action_matcher
                    ),
                },
            };
        }

        match self.config.default_effect {
            Effect::Allow => PolicyDecision {
                allowed: true,
                reason: "default allow".to_string(),
            },
            Effect::Deny => PolicyDecision {
                allowed: false,
                reason: "default deny".to_string(),
            },
        }
    }

    pub fn audit(
        &self,
        action: &str,
        context: &PolicyContext,
        decision: &PolicyDecision,
    ) -> Result<()> {
        if let Some(parent) = self.audit_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.audit_path)
            .with_context(|| {
                format!(
                    "failed to open policy audit file: {}",
                    self.audit_path.display()
                )
            })?;

        let line = serde_json::json!({
            "timestamp": Utc::now().to_rfc3339(),
            "action": action,
            "context": context,
            "allowed": decision.allowed,
            "reason": decision.reason,
        });
        writeln!(file, "{}", serde_json::to_string(&line)?)?;
        Ok(())
    }
}

fn match_action(rule: &PolicyRule, action: &str) -> bool {
    match rule.action_matcher {
        ActionMatcher::Exact => rule.action == action,
        ActionMatcher::Prefix => action.starts_with(rule.action.as_str()),
        ActionMatcher::Wildcard => wildcard_match(rule.action.as_str(), action),
        ActionMatcher::Regex => Regex::new(rule.action.as_str())
            .map(|re| re.is_match(action))
            .unwrap_or(false),
    }
}

fn match_scope(scope: &RuleScope, context: &PolicyContext) -> bool {
    if let Some(expected_source) = &scope.source {
        if context.source.as_deref() != Some(expected_source.as_str()) {
            return false;
        }
    }
    if let Some(expected_channel) = &scope.channel {
        if context.channel.as_deref() != Some(expected_channel.as_str()) {
            return false;
        }
    }
    if let Some(expected_agent) = &scope.agent {
        if context.agent.as_deref() != Some(expected_agent.as_str()) {
            return false;
        }
    }
    true
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p = pattern.as_bytes();
    let t = text.as_bytes();

    let mut i = 0usize;
    let mut j = 0usize;
    let mut star_idx: Option<usize> = None;
    let mut match_idx = 0usize;

    while j < t.len() {
        if i < p.len() && (p[i] == b'?' || p[i] == t[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == b'*' {
            star_idx = Some(i);
            match_idx = j;
            i += 1;
        } else if let Some(star_pos) = star_idx {
            i = star_pos + 1;
            match_idx += 1;
            j = match_idx;
        } else {
            return false;
        }
    }

    while i < p.len() && p[i] == b'*' {
        i += 1;
    }

    i == p.len()
}

pub fn load_policy_or_default(path: &Path) -> Result<PolicyConfig> {
    if !path.exists() {
        return Ok(PolicyConfig::default());
    }

    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read policy file: {}", path.display()))?;
    let cfg: PolicyConfig = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse policy json: {}", path.display()))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{
        ActionMatcher, Effect, PolicyConfig, PolicyContext, PolicyEngine, PolicyRule, RuleScope,
    };

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
    fn exact_rule_matches_backwards_compatible_shape() {
        let dir = temp_test_dir("policy_exact");
        let policy_path = dir.join("policy.json");
        let audit_path = dir.join("audit.jsonl");

        let cfg = PolicyConfig {
            default_effect: Effect::Allow,
            rules: vec![PolicyRule {
                effect: Effect::Deny,
                action: "tool:read_file".to_string(),
                action_matcher: ActionMatcher::Exact,
                scope: RuleScope::default(),
            }],
        };
        fs::write(
            &policy_path,
            serde_json::to_string_pretty(&cfg).expect("serialize policy should succeed"),
        )
        .expect("write policy should succeed");

        let engine =
            PolicyEngine::new(&policy_path, &audit_path).expect("engine should initialize");
        let decision = engine.evaluate("tool:read_file", &PolicyContext::default());
        assert!(!decision.allowed);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn wildcard_matcher_blocks_all_tool_actions() {
        let dir = temp_test_dir("policy_wildcard");
        let policy_path = dir.join("policy.json");
        let audit_path = dir.join("audit.jsonl");

        let cfg = PolicyConfig {
            default_effect: Effect::Allow,
            rules: vec![PolicyRule {
                effect: Effect::Deny,
                action: "tool:*".to_string(),
                action_matcher: ActionMatcher::Wildcard,
                scope: RuleScope::default(),
            }],
        };
        fs::write(
            &policy_path,
            serde_json::to_string_pretty(&cfg).expect("serialize policy should succeed"),
        )
        .expect("write policy should succeed");

        let engine =
            PolicyEngine::new(&policy_path, &audit_path).expect("engine should initialize");
        let decision = engine.evaluate("tool:echo", &PolicyContext::default());
        assert!(!decision.allowed);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn regex_matcher_and_scope_match_correctly() {
        let dir = temp_test_dir("policy_regex_scope");
        let policy_path = dir.join("policy.json");
        let audit_path = dir.join("audit.jsonl");

        let cfg = PolicyConfig {
            default_effect: Effect::Allow,
            rules: vec![PolicyRule {
                effect: Effect::Deny,
                action: "^dispatch:.*$".to_string(),
                action_matcher: ActionMatcher::Regex,
                scope: RuleScope {
                    source: Some("websocket".to_string()),
                    channel: None,
                    agent: Some("default".to_string()),
                },
            }],
        };
        fs::write(
            &policy_path,
            serde_json::to_string_pretty(&cfg).expect("serialize policy should succeed"),
        )
        .expect("write policy should succeed");

        let engine =
            PolicyEngine::new(&policy_path, &audit_path).expect("engine should initialize");

        let deny_decision = engine.evaluate(
            "dispatch:default",
            &PolicyContext::new(Some("websocket"), Some("websocket"), Some("default")),
        );
        assert!(!deny_decision.allowed);

        let allow_decision = engine.evaluate(
            "dispatch:default",
            &PolicyContext::new(Some("cli"), Some("cli"), Some("default")),
        );
        assert!(allow_decision.allowed);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn audit_writes_jsonl_record_with_context() {
        let dir = temp_test_dir("policy_audit");
        let policy_path = dir.join("policy.json");
        let audit_path = dir.join("audit.jsonl");

        let engine =
            PolicyEngine::new(&policy_path, &audit_path).expect("engine should initialize");
        let context = PolicyContext::new(Some("cli"), Some("cli"), Some("default"));
        let decision = engine.evaluate("tool:echo", &context);
        engine
            .audit("tool:echo", &context, &decision)
            .expect("audit should succeed");

        let content = fs::read_to_string(&audit_path).expect("audit file should exist");
        assert!(content.contains("\"action\":\"tool:echo\""));
        assert!(content.contains("\"context\""));

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }

    #[test]
    fn legacy_policy_json_without_matcher_or_scope_still_loads() {
        let dir = temp_test_dir("policy_legacy_json");
        let policy_path = dir.join("policy.json");
        let audit_path = dir.join("audit.jsonl");

        fs::write(
            &policy_path,
            r#"{"default_effect":"allow","rules":[{"effect":"deny","action":"tool:read_file"}]}"#,
        )
        .expect("write policy should succeed");

        let engine =
            PolicyEngine::new(&policy_path, &audit_path).expect("engine should initialize");
        let decision = engine.evaluate("tool:read_file", &PolicyContext::default());
        assert!(!decision.allowed);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
    }
}
