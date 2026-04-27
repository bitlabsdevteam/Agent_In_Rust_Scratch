use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::agent::types::Message;

pub struct SessionStore {
    path: PathBuf,
}

impl SessionStore {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn load(&self) -> Result<Vec<Message>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = fs::File::open(&self.path)
            .with_context(|| format!("failed to open session file: {}", self.path.display()))?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let message: Message = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse session line as json in {}",
                    self.path.display()
                )
            })?;
            messages.push(message);
        }

        Ok(messages)
    }

    pub fn append(&self, message: &Message) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create session directory: {}", parent.display())
            })?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| {
                format!("failed to open session for append: {}", self.path.display())
            })?;

        let line = serde_json::to_string(message)?;
        writeln!(file, "{line}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::SessionStore;
    use crate::agent::types::{Message, Role};

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
    fn load_returns_empty_when_session_file_missing() {
        let dir = temp_test_dir("session_missing");
        let store = SessionStore::new(dir.join("session.jsonl"));

        let loaded = store.load().expect("load should succeed");
        assert!(loaded.is_empty());

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn append_then_load_replays_messages_in_order() {
        let dir = temp_test_dir("session_roundtrip");
        let path = dir.join("session.jsonl");
        let store = SessionStore::new(&path);
        let first = Message::new(Role::User, "first");
        let second = Message::new(Role::Assistant, "second");

        store.append(&first).expect("append first should succeed");
        store.append(&second).expect("append second should succeed");

        let loaded = store.load().expect("load should succeed");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].content, "first");
        assert_eq!(loaded[1].content, "second");
        assert!(path.exists(), "session file should exist after append");

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn load_fails_on_malformed_jsonl_line() {
        let dir = temp_test_dir("session_malformed");
        let path = dir.join("session.jsonl");
        fs::write(&path, "{\"role\":\"User\",\"content\":\"ok\"}\nnot-json\n")
            .expect("failed to seed malformed session file");

        let store = SessionStore::new(path);
        let err = store
            .load()
            .expect_err("load should fail on malformed line");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("failed to parse session line as json"),
            "expected parse context, got: {rendered}"
        );

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }
}
