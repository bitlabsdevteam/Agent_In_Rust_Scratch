use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventSource {
    Cli,
    WebSocket,
    Telegram,
    Scheduler,
}

impl EventSource {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::WebSocket => "websocket",
            Self::Telegram => "telegram",
            Self::Scheduler => "scheduler",
        }
    }

    fn from_db(value: &str) -> Result<Self> {
        match value {
            "cli" => Ok(Self::Cli),
            "websocket" => Ok(Self::WebSocket),
            "telegram" => Ok(Self::Telegram),
            "scheduler" => Ok(Self::Scheduler),
            _ => Err(anyhow!("unsupported event source: {value}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeliveryTarget {
    Cli,
    File,
    WebSocket,
    Telegram,
}

impl DeliveryTarget {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Cli => "cli",
            Self::File => "file",
            Self::WebSocket => "websocket",
            Self::Telegram => "telegram",
        }
    }

    fn from_db(value: &str) -> Result<Self> {
        match value {
            "cli" => Ok(Self::Cli),
            "file" => Ok(Self::File),
            "websocket" => Ok(Self::WebSocket),
            "telegram" => Ok(Self::Telegram),
            _ => Err(anyhow!("unsupported delivery target: {value}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InboundPayload {
    UserMessage(String),
    WebSocketMessage { session_id: String, content: String },
    TelegramMessage { chat_id: i64, content: String },
    DispatchTask(DispatchTask),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DispatchTask {
    pub from_agent: String,
    pub to_agent: String,
    pub content: String,
}

impl DispatchTask {
    pub fn new(
        from_agent: impl Into<String>,
        to_agent: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            from_agent: from_agent.into(),
            to_agent: to_agent.into(),
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InboundEvent {
    pub source: EventSource,
    pub payload: InboundPayload,
}

impl InboundEvent {
    pub fn user_message(source: EventSource, content: impl Into<String>) -> Self {
        Self {
            source,
            payload: InboundPayload::UserMessage(content.into()),
        }
    }

    pub fn websocket_message(session_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            source: EventSource::WebSocket,
            payload: InboundPayload::WebSocketMessage {
                session_id: session_id.into(),
                content: content.into(),
            },
        }
    }

    pub fn telegram_message(chat_id: i64, content: impl Into<String>) -> Self {
        Self {
            source: EventSource::Telegram,
            payload: InboundPayload::TelegramMessage {
                chat_id,
                content: content.into(),
            },
        }
    }

    #[cfg(test)]
    pub fn dispatch_task(source: EventSource, task: DispatchTask) -> Self {
        Self {
            source,
            payload: InboundPayload::DispatchTask(task),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutboundEvent {
    pub target: DeliveryTarget,
    pub content: String,
    pub websocket_session_id: Option<String>,
    pub telegram_chat_id: Option<i64>,
}

impl OutboundEvent {
    pub fn new(target: DeliveryTarget, content: impl Into<String>) -> Self {
        Self {
            target,
            content: content.into(),
            websocket_session_id: None,
            telegram_chat_id: None,
        }
    }

    pub fn with_websocket_session(
        target: DeliveryTarget,
        content: impl Into<String>,
        websocket_session_id: impl Into<String>,
    ) -> Self {
        Self {
            target,
            content: content.into(),
            websocket_session_id: Some(websocket_session_id.into()),
            telegram_chat_id: None,
        }
    }

    pub fn with_telegram_chat(
        target: DeliveryTarget,
        content: impl Into<String>,
        telegram_chat_id: i64,
    ) -> Self {
        Self {
            target,
            content: content.into(),
            websocket_session_id: None,
            telegram_chat_id: Some(telegram_chat_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingOutbound {
    pub id: i64,
    pub event: OutboundEvent,
    pub attempts: i64,
}

type OutboundQueueRow = (i64, String, String, Option<String>, Option<i64>, i64);

#[derive(Debug)]
pub struct EventBus {
    conn: Connection,
}

impl EventBus {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create event bus directory: {}", parent.display())
            })?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open event database: {}", path.display()))?;
        let bus = Self { conn };
        bus.init_schema()?;
        Ok(bus)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS inbound_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT NULL
            );
            CREATE TABLE IF NOT EXISTS outbound_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                content_json TEXT NOT NULL,
                websocket_session_id TEXT NULL,
                telegram_chat_id INTEGER NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TEXT NULL,
                last_error TEXT NULL
            );
            CREATE TABLE IF NOT EXISTS websocket_sessions (
                session_id TEXT PRIMARY KEY,
                owner_agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS websocket_session_acl (
                session_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, agent_id)
            );
            CREATE TABLE IF NOT EXISTS telegram_chats (
                chat_id INTEGER PRIMARY KEY,
                owner_agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS telegram_chat_acl (
                chat_id INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (chat_id, agent_id)
            );",
        )?;
        self.migrate_outbound_events_schema()?;
        Ok(())
    }

    fn migrate_outbound_events_schema(&self) -> Result<()> {
        if !self.column_exists("outbound_events", "attempt_count")? {
            self.conn.execute(
                "ALTER TABLE outbound_events
                 ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0",
                [],
            )?;
        }
        if !self.column_exists("outbound_events", "last_attempt_at")? {
            self.conn.execute(
                "ALTER TABLE outbound_events
                 ADD COLUMN last_attempt_at TEXT NULL",
                [],
            )?;
        }
        if !self.column_exists("outbound_events", "last_error")? {
            self.conn.execute(
                "ALTER TABLE outbound_events
                 ADD COLUMN last_error TEXT NULL",
                [],
            )?;
        }
        if !self.column_exists("outbound_events", "websocket_session_id")? {
            self.conn.execute(
                "ALTER TABLE outbound_events
                 ADD COLUMN websocket_session_id TEXT NULL",
                [],
            )?;
        }
        if !self.column_exists("outbound_events", "telegram_chat_id")? {
            self.conn.execute(
                "ALTER TABLE outbound_events
                 ADD COLUMN telegram_chat_id INTEGER NULL",
                [],
            )?;
        }
        Ok(())
    }

    fn column_exists(&self, table: &str, column: &str) -> Result<bool> {
        let mut stmt = self
            .conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .with_context(|| format!("failed to inspect table schema: {table}"))?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn publish_inbound(&mut self, event: InboundEvent) -> Result<()> {
        let payload_json = serde_json::to_string(&event.payload)?;
        self.conn.execute(
            "INSERT INTO inbound_events (source, payload_json, created_at, processed_at)
             VALUES (?1, ?2, ?3, NULL)",
            params![event.source.as_str(), payload_json, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn pop_inbound(&mut self) -> Result<Option<InboundEvent>> {
        let tx = self.conn.transaction()?;
        let row: Option<(i64, String, String)> = tx
            .query_row(
                "SELECT id, source, payload_json
                 FROM inbound_events
                 WHERE processed_at IS NULL
                 ORDER BY id ASC
                 LIMIT 1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;

        let Some((id, source, payload_json)) = row else {
            tx.commit()?;
            return Ok(None);
        };

        tx.execute(
            "UPDATE inbound_events
             SET processed_at = ?1
             WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id],
        )?;
        tx.commit()?;

        let source = EventSource::from_db(&source)?;
        let payload: InboundPayload =
            serde_json::from_str(&payload_json).context("failed to decode inbound payload json")?;
        Ok(Some(InboundEvent { source, payload }))
    }

    pub fn publish_outbound(&mut self, event: OutboundEvent) -> Result<()> {
        let content_json = serde_json::to_string(&event.content)?;
        self.conn.execute(
            "INSERT INTO outbound_events (target, content_json, websocket_session_id, telegram_chat_id, created_at, processed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, NULL)",
            params![
                event.target.as_str(),
                content_json,
                event.websocket_session_id.as_deref(),
                event.telegram_chat_id,
                Utc::now().to_rfc3339()
            ],
        )?;
        Ok(())
    }

    pub fn next_outbound_for_delivery(&mut self) -> Result<Option<PendingOutbound>> {
        let row: Option<OutboundQueueRow> = self
            .conn
            .query_row(
                "SELECT id, target, content_json, websocket_session_id, telegram_chat_id, attempt_count
                 FROM outbound_events
                 WHERE processed_at IS NULL
                 ORDER BY id ASC
                 LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;

        let Some((id, target, content_json, websocket_session_id, telegram_chat_id, attempts)) =
            row
        else {
            return Ok(None);
        };

        let target = DeliveryTarget::from_db(&target)?;
        let content: String = serde_json::from_str(&content_json)
            .context("failed to decode outbound content json")?;
        Ok(Some(PendingOutbound {
            id,
            event: OutboundEvent {
                target,
                content,
                websocket_session_id,
                telegram_chat_id,
            },
            attempts,
        }))
    }

    pub fn mark_outbound_processed(&mut self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE outbound_events
             SET processed_at = ?1
             WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id],
        )?;
        Ok(())
    }

    pub fn mark_outbound_delivery_failed(&mut self, id: i64, err: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE outbound_events
             SET attempt_count = attempt_count + 1,
                 last_attempt_at = ?1,
                 last_error = ?2
             WHERE id = ?3",
            params![Utc::now().to_rfc3339(), err, id],
        )?;
        Ok(())
    }

    pub fn inbound_pending_len(&self) -> Result<usize> {
        let count = self.conn.query_row(
            "SELECT COUNT(*) FROM inbound_events WHERE processed_at IS NULL",
            [],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    pub fn upsert_websocket_session_owner(
        &self,
        session_id: &str,
        owner_agent_id: &str,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO websocket_sessions (session_id, owner_agent_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?3)
             ON CONFLICT(session_id)
             DO UPDATE SET owner_agent_id = excluded.owner_agent_id, updated_at = excluded.updated_at",
            params![session_id, owner_agent_id, now],
        )?;
        Ok(())
    }

    pub fn websocket_session_owner(&self, session_id: &str) -> Result<Option<String>> {
        let owner = self
            .conn
            .query_row(
                "SELECT owner_agent_id FROM websocket_sessions WHERE session_id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(owner)
    }

    pub fn allow_websocket_session_agent(&self, session_id: &str, agent_id: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO websocket_session_acl (session_id, agent_id, created_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(session_id, agent_id)
             DO NOTHING",
            params![session_id, agent_id, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn websocket_session_acl(&self, session_id: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT agent_id
             FROM websocket_session_acl
             WHERE session_id = ?1
             ORDER BY agent_id ASC",
        )?;
        let mut rows = stmt.query(params![session_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row.get(0)?);
        }
        Ok(out)
    }

    pub fn websocket_session_agent_allowed(
        &self,
        session_id: &str,
        agent_id: &str,
    ) -> Result<bool> {
        if let Some(owner) = self.websocket_session_owner(session_id)? {
            if owner == agent_id {
                return Ok(true);
            }
        }
        let exists: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1
                 FROM websocket_session_acl
                 WHERE session_id = ?1 AND agent_id = ?2
                 LIMIT 1",
                params![session_id, agent_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    pub fn upsert_telegram_chat_owner(&self, chat_id: i64, owner_agent_id: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO telegram_chats (chat_id, owner_agent_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?3)
             ON CONFLICT(chat_id)
             DO UPDATE SET owner_agent_id = excluded.owner_agent_id, updated_at = excluded.updated_at",
            params![chat_id, owner_agent_id, now],
        )?;
        Ok(())
    }

    pub fn telegram_chat_owner(&self, chat_id: i64) -> Result<Option<String>> {
        let owner = self
            .conn
            .query_row(
                "SELECT owner_agent_id FROM telegram_chats WHERE chat_id = ?1",
                params![chat_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(owner)
    }

    pub fn allow_telegram_chat_agent(&self, chat_id: i64, agent_id: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO telegram_chat_acl (chat_id, agent_id, created_at)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(chat_id, agent_id)
             DO NOTHING",
            params![chat_id, agent_id, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    pub fn telegram_chat_acl(&self, chat_id: i64) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT agent_id
             FROM telegram_chat_acl
             WHERE chat_id = ?1
             ORDER BY agent_id ASC",
        )?;
        let mut rows = stmt.query(params![chat_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row.get(0)?);
        }
        Ok(out)
    }

    pub fn telegram_chat_agent_allowed(&self, chat_id: i64, agent_id: &str) -> Result<bool> {
        if let Some(owner) = self.telegram_chat_owner(chat_id)? {
            if owner == agent_id {
                return Ok(true);
            }
        }
        let exists: Option<i64> = self
            .conn
            .query_row(
                "SELECT 1
                 FROM telegram_chat_acl
                 WHERE chat_id = ?1 AND agent_id = ?2
                 LIMIT 1",
                params![chat_id, agent_id],
                |row| row.get(0),
            )
            .optional()?;
        Ok(exists.is_some())
    }

    #[cfg(test)]
    pub fn pop_outbound(&mut self) -> Result<Option<OutboundEvent>> {
        let Some(pending) = self.next_outbound_for_delivery()? else {
            return Ok(None);
        };
        self.mark_outbound_processed(pending.id)?;
        Ok(Some(pending.event))
    }

    #[cfg(test)]
    pub fn outbound_pending_len(&self) -> usize {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM outbound_events WHERE processed_at IS NULL",
                [],
                |row| row.get(0),
            )
            .expect("outbound count query should succeed")
    }

    #[cfg(test)]
    pub fn outbound_attempt_count(&self, id: i64) -> i64 {
        self.conn
            .query_row(
                "SELECT attempt_count FROM outbound_events WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .expect("outbound attempt query should succeed")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use anyhow::Result;
    use rusqlite::Connection;

    use super::{DeliveryTarget, DispatchTask, EventBus, EventSource, InboundEvent, OutboundEvent};

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
    fn event_bus_preserves_inbound_fifo_order() -> Result<()> {
        let dir = temp_test_dir("event_inbound_fifo");
        let mut bus = EventBus::open(dir.join("events.db"))?;
        bus.publish_inbound(InboundEvent::user_message(EventSource::Cli, "a"))?;
        bus.publish_inbound(InboundEvent::user_message(EventSource::Cli, "b"))?;

        let first = bus
            .pop_inbound()?
            .expect("first inbound event should exist");
        let second = bus
            .pop_inbound()?
            .expect("second inbound event should exist");

        assert_eq!(first, InboundEvent::user_message(EventSource::Cli, "a"));
        assert_eq!(second, InboundEvent::user_message(EventSource::Cli, "b"));
        assert!(bus.pop_inbound()?.is_none());
        assert_eq!(bus.inbound_pending_len()?, 0);

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn event_bus_preserves_outbound_fifo_order() -> Result<()> {
        let dir = temp_test_dir("event_outbound_fifo");
        let mut bus = EventBus::open(dir.join("events.db"))?;
        bus.publish_outbound(OutboundEvent::new(DeliveryTarget::Cli, "first"))?;
        bus.publish_outbound(OutboundEvent::new(DeliveryTarget::Cli, "second"))?;

        let first = bus
            .pop_outbound()?
            .expect("first outbound event should exist");
        let second = bus
            .pop_outbound()?
            .expect("second outbound event should exist");

        assert_eq!(first.content, "first");
        assert_eq!(second.content, "second");
        assert!(bus.pop_outbound()?.is_none());
        assert_eq!(bus.outbound_pending_len(), 0);

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn reopen_recovers_unprocessed_rows() -> Result<()> {
        let dir = temp_test_dir("event_reopen");
        let db = dir.join("events.db");

        {
            let mut bus = EventBus::open(&db)?;
            bus.publish_inbound(InboundEvent::user_message(EventSource::Cli, "first"))?;
            bus.publish_inbound(InboundEvent::user_message(EventSource::Cli, "second"))?;
            let popped = bus.pop_inbound()?.expect("one inbound should pop");
            assert_eq!(
                popped,
                InboundEvent::user_message(EventSource::Cli, "first")
            );
            assert_eq!(bus.inbound_pending_len()?, 1);
        }

        {
            let mut reopened = EventBus::open(&db)?;
            let next = reopened
                .pop_inbound()?
                .expect("second inbound should remain unprocessed");
            assert_eq!(next, InboundEvent::user_message(EventSource::Cli, "second"));
            assert!(reopened.pop_inbound()?.is_none());
        }

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn pop_inbound_fails_when_payload_json_is_malformed() -> Result<()> {
        let dir = temp_test_dir("event_malformed");
        let db = dir.join("events.db");
        let mut bus = EventBus::open(&db)?;

        let conn = Connection::open(&db)?;
        conn.execute(
            "INSERT INTO inbound_events (source, payload_json, created_at, processed_at)
             VALUES ('cli', 'not-json', '2026-01-01T00:00:00Z', NULL)",
            [],
        )?;

        let err = bus
            .pop_inbound()
            .expect_err("pop should fail on malformed payload");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("failed to decode inbound payload json"),
            "expected decode context, got: {rendered}"
        );

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn dispatch_task_payload_roundtrips_through_sqlite_queue() -> Result<()> {
        let dir = temp_test_dir("event_dispatch_roundtrip");
        let mut bus = EventBus::open(dir.join("events.db"))?;
        let task = DispatchTask::new("default", "support-agent", "triage this issue");
        bus.publish_inbound(InboundEvent::dispatch_task(EventSource::Cli, task.clone()))?;

        let popped = bus.pop_inbound()?.expect("dispatch task should pop");
        assert_eq!(popped, InboundEvent::dispatch_task(EventSource::Cli, task));

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn open_migrates_legacy_outbound_events_schema_without_attempt_columns() -> Result<()> {
        let dir = temp_test_dir("event_legacy_migration");
        let db = dir.join("events.db");

        let conn = Connection::open(&db)?;
        conn.execute_batch(
            "CREATE TABLE inbound_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT NULL
            );
            CREATE TABLE outbound_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                content_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT NULL
            );
            INSERT INTO outbound_events (target, content_json, created_at, processed_at)
            VALUES ('cli', '\"legacy\"', '2026-01-01T00:00:00Z', NULL);",
        )?;
        drop(conn);

        let mut bus = EventBus::open(&db)?;
        let pending = bus
            .next_outbound_for_delivery()?
            .expect("legacy row should remain readable after migration");
        assert_eq!(pending.event.content, "legacy");
        assert_eq!(pending.attempts, 0);

        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn websocket_session_owner_persists_across_reopen() -> Result<()> {
        let dir = temp_test_dir("ws_owner_reopen");
        let db = dir.join("events.db");
        {
            let bus = EventBus::open(&db)?;
            bus.upsert_websocket_session_owner("sess-1", "default")?;
        }
        {
            let bus = EventBus::open(&db)?;
            let owner = bus.websocket_session_owner("sess-1")?;
            assert_eq!(owner.as_deref(), Some("default"));
        }
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn websocket_session_acl_controls_agent_access() -> Result<()> {
        let dir = temp_test_dir("ws_acl");
        let db = dir.join("events.db");
        let bus = EventBus::open(&db)?;
        bus.upsert_websocket_session_owner("sess-2", "owner-agent")?;
        assert!(bus.websocket_session_agent_allowed("sess-2", "owner-agent")?);
        assert!(!bus.websocket_session_agent_allowed("sess-2", "other-agent")?);
        bus.allow_websocket_session_agent("sess-2", "other-agent")?;
        assert!(bus.websocket_session_agent_allowed("sess-2", "other-agent")?);
        let acl = bus.websocket_session_acl("sess-2")?;
        assert_eq!(acl, vec!["other-agent".to_string()]);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn telegram_chat_owner_persists_across_reopen() -> Result<()> {
        let dir = temp_test_dir("tg_owner_reopen");
        let db = dir.join("events.db");
        {
            let bus = EventBus::open(&db)?;
            bus.upsert_telegram_chat_owner(1001, "default")?;
        }
        {
            let bus = EventBus::open(&db)?;
            let owner = bus.telegram_chat_owner(1001)?;
            assert_eq!(owner.as_deref(), Some("default"));
        }
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }

    #[test]
    fn telegram_chat_acl_controls_agent_access() -> Result<()> {
        let dir = temp_test_dir("tg_acl");
        let db = dir.join("events.db");
        let bus = EventBus::open(&db)?;
        bus.upsert_telegram_chat_owner(1002, "owner-agent")?;
        assert!(bus.telegram_chat_agent_allowed(1002, "owner-agent")?);
        assert!(!bus.telegram_chat_agent_allowed(1002, "other-agent")?);
        bus.allow_telegram_chat_agent(1002, "other-agent")?;
        assert!(bus.telegram_chat_agent_allowed(1002, "other-agent")?);
        let acl = bus.telegram_chat_acl(1002)?;
        assert_eq!(acl, vec!["other-agent".to_string()]);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
        Ok(())
    }
}
