use std::cmp::Ordering;
use std::collections::{HashSet, hash_map::DefaultHasher};
use std::env;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use reqwest::blocking::Client;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Deserialize;

use crate::agent::types::{Message, Role};

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecord {
    pub id: i64,
    pub summary: String,
    pub created_at: String,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCompactionStats {
    pub removed_ttl: usize,
    pub removed_overflow: usize,
    pub remaining: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAnnMaintenanceStats {
    pub scanned_rows: usize,
    pub backfilled_rows: usize,
    pub reindexed_rows: usize,
    pub remaining_unindexed: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDiagnostics {
    pub search_total: u64,
    pub ann_attempt_total: u64,
    pub ann_hit_total: u64,
    pub ann_fallback_total: u64,
    pub scan_only_total: u64,
    pub ann_hit_rate: f64,
    pub ann_fallback_rate: f64,
    pub scan_only_rate: f64,
    pub candidate_examined_total: u64,
    pub candidate_examined_avg: f64,
    pub candidate_examined_p95: u64,
    pub query_latency_avg_ms: f64,
    pub query_latency_p95_ms: u64,
    pub ann_cursor: i64,
    pub ann_cursor_max: i64,
    pub ann_cursor_progress: f64,
    pub remaining_unindexed: usize,
    pub ann_last_maintained_at: Option<String>,
}

type MemoryRow = (i64, String, String, Option<String>);
type MemoryScanRow = (i64, String, Option<String>);

const LATENCY_BUCKET_BOUNDS_MS: &[u64] = &[1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000];
const CANDIDATE_BUCKET_BOUNDS: &[u64] = &[8, 16, 32, 64, 128, 256, 512, 1_024];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySearchMode {
    Lexical,
    Semantic,
    Hybrid,
}

impl MemorySearchMode {
    fn from_raw(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "lexical" => Self::Lexical,
            "semantic" => Self::Semantic,
            "hybrid" => Self::Hybrid,
            _ => Self::Hybrid,
        }
    }
}

pub struct MemoryWorker {
    conn: Connection,
    snapshot_window: usize,
    retention_ttl_days: i64,
    retention_max_records: usize,
    search_mode: MemorySearchMode,
    hybrid_lexical_weight: f64,
    hybrid_semantic_weight: f64,
    embedding_dimensions: usize,
    embedding_backend: EmbeddingBackend,
    ann_enabled: bool,
    ann_scan_threshold: usize,
    ann_probe_count: usize,
    ann_candidate_cap: usize,
}

#[derive(Debug, Clone)]
enum EmbeddingBackend {
    LocalHashed,
    OpenAI(OpenAIEmbeddingClient),
}

#[derive(Debug, Clone)]
struct OpenAIEmbeddingClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl MemoryWorker {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        let worker = Self {
            conn,
            snapshot_window: 8,
            retention_ttl_days: 30,
            retention_max_records: 2000,
            search_mode: MemorySearchMode::Hybrid,
            hybrid_lexical_weight: 0.6,
            hybrid_semantic_weight: 0.4,
            embedding_dimensions: 128,
            embedding_backend: EmbeddingBackend::LocalHashed,
            ann_enabled: true,
            ann_scan_threshold: 200,
            ann_probe_count: 6,
            ann_candidate_cap: 256,
        };
        worker.init_schema()?;
        Ok(worker)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding_json TEXT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_ann_index (
                memory_id INTEGER NOT NULL,
                bucket_key TEXT NOT NULL,
                PRIMARY KEY(memory_id, bucket_key),
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_memory_ann_bucket
              ON memory_ann_index(bucket_key)
            ;
            CREATE INDEX IF NOT EXISTS idx_memory_ann_memory
              ON memory_ann_index(memory_id)
            ;",
        )?;
        if !self.column_exists("memories", "embedding_json")? {
            self.conn.execute(
                "ALTER TABLE memories
                 ADD COLUMN embedding_json TEXT NULL",
                [],
            )?;
        }
        Ok(())
    }

    pub fn set_retention(&mut self, ttl_days: i64, max_records: usize) {
        self.retention_ttl_days = ttl_days.max(1);
        self.retention_max_records = max_records.max(1);
    }

    pub fn set_search_profile(
        &mut self,
        mode_raw: &str,
        hybrid_lexical_weight: f64,
        hybrid_semantic_weight: f64,
    ) {
        self.search_mode = MemorySearchMode::from_raw(mode_raw);

        let lexical = if hybrid_lexical_weight.is_finite() {
            hybrid_lexical_weight.max(0.0)
        } else {
            0.6
        };
        let semantic = if hybrid_semantic_weight.is_finite() {
            hybrid_semantic_weight.max(0.0)
        } else {
            0.4
        };
        if lexical == 0.0 && semantic == 0.0 {
            self.hybrid_lexical_weight = 0.6;
            self.hybrid_semantic_weight = 0.4;
        } else {
            self.hybrid_lexical_weight = lexical;
            self.hybrid_semantic_weight = semantic;
        }
    }

    pub fn set_embedding_backend(
        &mut self,
        provider: &str,
        model: &str,
        base_url: &str,
        api_key_env: &str,
        dimensions: usize,
    ) {
        self.embedding_dimensions = dimensions.max(16);
        let provider = provider.trim().to_ascii_lowercase();

        if provider != "openai" {
            self.embedding_backend = EmbeddingBackend::LocalHashed;
            return;
        }

        let env_name = if api_key_env.trim().is_empty() {
            "OPENAI_API_KEY"
        } else {
            api_key_env.trim()
        };
        let api_key = env::var(env_name).unwrap_or_default();
        if api_key.trim().is_empty() {
            self.embedding_backend = EmbeddingBackend::LocalHashed;
            return;
        }

        let embed_model = if model.trim().is_empty() {
            "text-embedding-3-small".to_string()
        } else {
            model.trim().to_string()
        };
        let embed_base_url = if base_url.trim().is_empty() {
            "https://api.openai.com/v1".to_string()
        } else {
            base_url.trim().trim_end_matches('/').to_string()
        };
        self.embedding_backend = EmbeddingBackend::OpenAI(OpenAIEmbeddingClient {
            client: Client::new(),
            api_key,
            base_url: embed_base_url,
            model: embed_model,
        });
    }

    pub fn set_ann_profile(
        &mut self,
        enabled: bool,
        scan_threshold: usize,
        probe_count: usize,
        candidate_cap: usize,
    ) {
        self.ann_enabled = enabled;
        self.ann_scan_threshold = scan_threshold.max(32);
        self.ann_probe_count = probe_count.max(1);
        self.ann_candidate_cap = candidate_cap.max(32);
    }

    pub fn maybe_snapshot(&mut self, history: &[Message]) -> Result<Option<i64>> {
        let last_index = self.last_snapshot_index()?;
        if history.len() < last_index.saturating_add(self.snapshot_window) {
            return Ok(None);
        }

        let start = last_index;
        let end = history.len();
        let summary = summarize_messages(&history[start..end]);
        let embedding = self.embed(summary.as_str());
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO memories (summary, created_at, embedding_json) VALUES (?1, ?2, ?3)",
            params![summary, now, vector_to_json(embedding.as_slice())],
        )?;
        let id = self.conn.last_insert_rowid();
        self.upsert_ann_for_memory(id, embedding.as_slice())?;
        self.set_last_snapshot_index(end)?;
        let _ = self.compact()?;
        Ok(Some(id))
    }

    pub fn latest(&self, limit: usize) -> Result<Vec<MemoryRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, summary, created_at
             FROM memories
             ORDER BY id DESC
             LIMIT ?1",
        )?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(MemoryRecord {
                id: row.get(0)?,
                summary: row.get(1)?,
                created_at: row.get(2)?,
                score: 0.0,
            });
        }
        Ok(out)
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryRecord>> {
        let q = query.trim();
        if q.is_empty() {
            return Ok(Vec::new());
        }
        let started_at = Instant::now();
        self.increment_meta_counter("search_total", 1)?;
        let q_embedding = self.embed(q);

        let total: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

        let use_ann = self.ann_enabled
            && total >= self.ann_scan_threshold
            && !matches!(self.search_mode, MemorySearchMode::Lexical);

        let rows_buf = if use_ann {
            self.increment_meta_counter("ann_attempt_total", 1)?;
            let keys = ann_bucket_keys(q_embedding.as_slice(), self.ann_probe_count);
            let ann_rows = self.fetch_ann_candidates(keys.as_slice(), self.ann_candidate_cap)?;
            if ann_rows.is_empty() {
                self.increment_meta_counter("ann_fallback_total", 1)?;
                self.increment_meta_counter("search_path_fallback_scan_total", 1)?;
                self.fetch_recent_rows(256)?
            } else {
                self.increment_meta_counter("ann_hit_total", 1)?;
                self.increment_meta_counter("search_path_ann_hit_total", 1)?;
                ann_rows
            }
        } else {
            self.increment_meta_counter("search_path_scan_only_total", 1)?;
            self.fetch_recent_rows(256)?
        };
        let candidate_count = rows_buf.len() as u64;
        self.increment_meta_counter("candidate_examined_total", candidate_count)?;
        self.observe_histogram_u64(
            "candidate_examined_hist",
            CANDIDATE_BUCKET_BOUNDS,
            candidate_count,
        )?;

        let mut candidates = Vec::new();
        for (id, summary, created_at, embedding_json) in rows_buf {
            let lexical_score = lexical_relevance(q, &summary);
            let memory_embedding = embedding_json
                .as_deref()
                .and_then(parse_vector_json)
                .unwrap_or_else(|| embed_text(summary.as_str(), self.embedding_dimensions));
            let semantic_score =
                semantic_similarity(q_embedding.as_slice(), memory_embedding.as_slice());
            let recency_score = recency_boost(created_at.as_str());
            let score = rank_memory(
                self.search_mode,
                self.hybrid_lexical_weight,
                self.hybrid_semantic_weight,
                lexical_score,
                semantic_score,
                recency_score,
            );
            if score > 0.0 {
                candidates.push(MemoryRecord {
                    id,
                    summary,
                    created_at,
                    score,
                });
            }
        }

        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.id.cmp(&a.id))
        });
        candidates.truncate(limit);
        let elapsed = started_at.elapsed();
        let elapsed_ms = elapsed.as_millis() as u64;
        let elapsed_us = elapsed.as_micros() as u64;
        self.increment_meta_counter("query_latency_total_us", elapsed_us)?;
        self.observe_histogram_u64(
            "query_latency_hist_ms",
            LATENCY_BUCKET_BOUNDS_MS,
            elapsed_ms,
        )?;
        Ok(candidates)
    }

    pub fn compact(&mut self) -> Result<MemoryCompactionStats> {
        let cutoff = Utc::now() - Duration::days(self.retention_ttl_days);
        let cutoff_rfc = cutoff.to_rfc3339();

        let removed_ttl = self.conn.execute(
            "DELETE FROM memories WHERE created_at < ?1",
            params![cutoff_rfc],
        )?;

        let total_after_ttl: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;

        let mut removed_overflow = 0usize;
        if total_after_ttl > self.retention_max_records {
            let overflow = total_after_ttl - self.retention_max_records;
            removed_overflow = self.conn.execute(
                "DELETE FROM memories
                 WHERE id IN (
                    SELECT id FROM memories
                    ORDER BY id ASC
                    LIMIT ?1
                 )",
                params![overflow as i64],
            )?;
        }

        let remaining: usize = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))?;
        self.conn.execute(
            "DELETE FROM memory_ann_index
             WHERE memory_id NOT IN (SELECT id FROM memories)",
            [],
        )?;

        Ok(MemoryCompactionStats {
            removed_ttl,
            removed_overflow,
            remaining,
        })
    }

    pub fn maintain_ann_index(&mut self, batch_size: usize) -> Result<MemoryAnnMaintenanceStats> {
        let batch = batch_size.max(1);
        let mut cursor = self.get_meta_i64("ann_maint_cursor")?.unwrap_or(0);
        let mut rows = self.fetch_rows_after(cursor, batch)?;
        if rows.is_empty() {
            cursor = 0;
            rows = self.fetch_rows_after(cursor, batch)?;
        }

        let mut scanned = 0usize;
        let mut backfilled = 0usize;
        let mut reindexed = 0usize;
        let mut last_seen_id = cursor;

        for (id, summary, embedding_json) in rows {
            scanned += 1;
            last_seen_id = id;

            let had_ann = self.has_ann_index(id)?;
            let stored_embedding = embedding_json.as_deref().and_then(parse_vector_json);
            let valid_embedding = stored_embedding
                .as_ref()
                .filter(|v| v.len() == self.embedding_dimensions)
                .cloned();

            let embedding = valid_embedding
                .clone()
                .unwrap_or_else(|| self.embed(summary.as_str()));
            let needs_embedding_update = valid_embedding.is_none();
            let needs_ann_update = !had_ann || needs_embedding_update;

            if needs_embedding_update {
                self.conn.execute(
                    "UPDATE memories SET embedding_json = ?1 WHERE id = ?2",
                    params![vector_to_json(embedding.as_slice()), id],
                )?;
            }

            if needs_ann_update {
                self.conn.execute(
                    "DELETE FROM memory_ann_index WHERE memory_id = ?1",
                    params![id],
                )?;
                self.upsert_ann_for_memory(id, embedding.as_slice())?;
                if had_ann {
                    reindexed += 1;
                } else {
                    backfilled += 1;
                }
            }
        }

        self.set_meta_i64("ann_maint_cursor", last_seen_id)?;
        let remaining_unindexed = self.count_remaining_unindexed()?;
        let now = Utc::now().to_rfc3339();
        self.set_meta_text("ann_maint_last_run_at", now.as_str())?;

        Ok(MemoryAnnMaintenanceStats {
            scanned_rows: scanned,
            backfilled_rows: backfilled,
            reindexed_rows: reindexed,
            remaining_unindexed,
        })
    }

    pub fn diagnostics(&self) -> Result<MemoryDiagnostics> {
        let search_total = self.get_meta_u64("search_total")?.unwrap_or(0);
        let ann_attempt_total = self.get_meta_u64("ann_attempt_total")?.unwrap_or(0);
        let ann_hit_total = self.get_meta_u64("ann_hit_total")?.unwrap_or(0);
        let ann_fallback_total = self.get_meta_u64("ann_fallback_total")?.unwrap_or(0);
        let scan_only_total = self
            .get_meta_u64("search_path_scan_only_total")?
            .unwrap_or(0);
        let candidate_examined_total = self.get_meta_u64("candidate_examined_total")?.unwrap_or(0);
        let query_latency_total_us = self.get_meta_u64("query_latency_total_us")?.unwrap_or(0);

        let ann_hit_rate = if ann_attempt_total == 0 {
            0.0
        } else {
            ann_hit_total as f64 / ann_attempt_total as f64
        };
        let ann_fallback_rate = if ann_attempt_total == 0 {
            0.0
        } else {
            ann_fallback_total as f64 / ann_attempt_total as f64
        };
        let scan_only_rate = if search_total == 0 {
            0.0
        } else {
            scan_only_total as f64 / search_total as f64
        };
        let candidate_examined_avg = if search_total == 0 {
            0.0
        } else {
            candidate_examined_total as f64 / search_total as f64
        };
        let query_latency_avg_ms = if search_total == 0 {
            0.0
        } else {
            (query_latency_total_us as f64 / search_total as f64) / 1000.0
        };

        let ann_cursor = self.get_meta_i64("ann_maint_cursor")?.unwrap_or(0);
        let ann_cursor_max = self.max_memory_id()?;
        let ann_cursor_progress = if ann_cursor_max <= 0 {
            1.0
        } else {
            (ann_cursor.clamp(0, ann_cursor_max) as f64) / (ann_cursor_max as f64)
        };

        Ok(MemoryDiagnostics {
            search_total,
            ann_attempt_total,
            ann_hit_total,
            ann_fallback_total,
            scan_only_total,
            ann_hit_rate,
            ann_fallback_rate,
            scan_only_rate,
            candidate_examined_total,
            candidate_examined_avg,
            candidate_examined_p95: self.estimate_histogram_p95_u64(
                "candidate_examined_hist",
                CANDIDATE_BUCKET_BOUNDS,
                search_total,
            )?,
            query_latency_avg_ms,
            query_latency_p95_ms: self.estimate_histogram_p95_u64(
                "query_latency_hist_ms",
                LATENCY_BUCKET_BOUNDS_MS,
                search_total,
            )?,
            ann_cursor,
            ann_cursor_max,
            ann_cursor_progress,
            remaining_unindexed: self.count_remaining_unindexed()?,
            ann_last_maintained_at: self.get_meta_text("ann_maint_last_run_at")?,
        })
    }

    fn last_snapshot_index(&self) -> Result<usize> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM memory_meta WHERE key = 'last_snapshot_index'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        Ok(raw.and_then(|v| v.parse::<usize>().ok()).unwrap_or(0))
    }

    fn set_last_snapshot_index(&self, index: usize) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_meta (key, value)
             VALUES ('last_snapshot_index', ?1)
             ON CONFLICT(key)
             DO UPDATE SET value = excluded.value",
            params![index.to_string()],
        )?;
        Ok(())
    }

    fn get_meta_i64(&self, key: &str) -> Result<Option<i64>> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM memory_meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(raw.and_then(|v| v.parse::<i64>().ok()))
    }

    fn set_meta_i64(&self, key: &str, value: i64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_meta (key, value)
             VALUES (?1, ?2)
             ON CONFLICT(key)
             DO UPDATE SET value = excluded.value",
            params![key, value.to_string()],
        )?;
        Ok(())
    }

    fn get_meta_text(&self, key: &str) -> Result<Option<String>> {
        let raw: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM memory_meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        Ok(raw)
    }

    fn set_meta_text(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_meta (key, value)
             VALUES (?1, ?2)
             ON CONFLICT(key)
             DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    fn get_meta_u64(&self, key: &str) -> Result<Option<u64>> {
        Ok(self
            .get_meta_i64(key)?
            .and_then(|v| if v < 0 { None } else { Some(v as u64) }))
    }

    fn increment_meta_counter(&self, key: &str, delta: u64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_meta (key, value)
             VALUES (?1, ?2)
             ON CONFLICT(key)
             DO UPDATE SET value = CAST(memory_meta.value AS INTEGER) + CAST(excluded.value AS INTEGER)",
            params![key, delta as i64],
        )?;
        Ok(())
    }

    fn observe_histogram_u64(&self, key_prefix: &str, bounds: &[u64], value: u64) -> Result<()> {
        let bucket_idx = histogram_bucket_index(bounds, value);
        let key = if bucket_idx >= bounds.len() {
            format!("{key_prefix}_overflow")
        } else {
            format!("{key_prefix}_{bucket_idx}")
        };
        self.increment_meta_counter(key.as_str(), 1)
    }

    fn estimate_histogram_p95_u64(
        &self,
        key_prefix: &str,
        bounds: &[u64],
        sample_total: u64,
    ) -> Result<u64> {
        if sample_total == 0 {
            return Ok(0);
        }
        let target = ((sample_total as f64) * 0.95).ceil() as u64;
        let mut seen = 0u64;

        for (idx, bound) in bounds.iter().enumerate() {
            let key = format!("{key_prefix}_{idx}");
            seen = seen.saturating_add(self.get_meta_u64(key.as_str())?.unwrap_or(0));
            if seen >= target {
                return Ok(*bound);
            }
        }

        let overflow_key = format!("{key_prefix}_overflow");
        seen = seen.saturating_add(self.get_meta_u64(overflow_key.as_str())?.unwrap_or(0));
        if seen >= target {
            return Ok(*bounds.last().unwrap_or(&0));
        }
        Ok(*bounds.last().unwrap_or(&0))
    }

    fn max_memory_id(&self) -> Result<i64> {
        let max: i64 =
            self.conn
                .query_row("SELECT COALESCE(MAX(id), 0) FROM memories", [], |row| {
                    row.get(0)
                })?;
        Ok(max)
    }

    fn count_remaining_unindexed(&self) -> Result<usize> {
        let remaining_unindexed: usize = self.conn.query_row(
            "SELECT COUNT(*)
             FROM memories m
             WHERE NOT EXISTS (
                 SELECT 1 FROM memory_ann_index idx WHERE idx.memory_id = m.id
             )",
            [],
            |row| row.get(0),
        )?;
        Ok(remaining_unindexed)
    }

    fn column_exists(&self, table: &str, column: &str) -> Result<bool> {
        let pragma = format!("PRAGMA table_info({table})");
        let mut stmt = self.conn.prepare(&pragma)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let name: String = row.get(1)?;
            if name == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn embed(&self, input: &str) -> Vec<f64> {
        match &self.embedding_backend {
            EmbeddingBackend::LocalHashed => embed_text(input, self.embedding_dimensions),
            EmbeddingBackend::OpenAI(client) => client
                .embed(input)
                .unwrap_or_else(|_| embed_text(input, self.embedding_dimensions)),
        }
    }

    fn fetch_recent_rows(&self, cap: usize) -> Result<Vec<MemoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, summary, created_at, embedding_json
             FROM memories
             ORDER BY id DESC
             LIMIT ?1",
        )?;
        let mut rows = stmt.query(params![cap as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?));
        }
        Ok(out)
    }

    fn fetch_rows_after(&self, after_id: i64, cap: usize) -> Result<Vec<MemoryScanRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, summary, embedding_json
             FROM memories
             WHERE id > ?1
             ORDER BY id ASC
             LIMIT ?2",
        )?;
        let mut rows = stmt.query(params![after_id, cap as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((row.get(0)?, row.get(1)?, row.get(2)?));
        }
        Ok(out)
    }

    fn has_ann_index(&self, memory_id: i64) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memory_ann_index WHERE memory_id = ?1",
            params![memory_id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    fn fetch_ann_candidates(&self, keys: &[String], cap: usize) -> Result<Vec<MemoryRow>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let mut dedup = HashSet::new();
        let unique_keys: Vec<String> = keys
            .iter()
            .filter_map(|k| {
                if dedup.insert(k.clone()) {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();
        if unique_keys.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders = (0..unique_keys.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT DISTINCT m.id, m.summary, m.created_at, m.embedding_json
             FROM memories m
             JOIN memory_ann_index idx ON idx.memory_id = m.id
             WHERE idx.bucket_key IN ({})
             ORDER BY m.id DESC
             LIMIT {}",
            placeholders, cap
        );
        let mut stmt = self.conn.prepare(sql.as_str())?;
        let mut rows = stmt.query(rusqlite::params_from_iter(unique_keys.iter()))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?));
        }
        Ok(out)
    }

    fn upsert_ann_for_memory(&self, memory_id: i64, embedding: &[f64]) -> Result<()> {
        let keys = ann_bucket_keys(embedding, self.ann_probe_count);
        if keys.is_empty() {
            return Ok(());
        }
        for key in keys {
            self.conn.execute(
                "INSERT INTO memory_ann_index (memory_id, bucket_key)
                 VALUES (?1, ?2)
                 ON CONFLICT(memory_id, bucket_key) DO NOTHING",
                params![memory_id, key],
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn insert_test_record(&self, summary: &str, created_at: &str) -> Result<()> {
        let embedding = embed_text(summary, self.embedding_dimensions);
        self.conn.execute(
            "INSERT INTO memories (summary, created_at, embedding_json) VALUES (?1, ?2, ?3)",
            params![summary, created_at, vector_to_json(embedding.as_slice())],
        )?;
        let id = self.conn.last_insert_rowid();
        self.upsert_ann_for_memory(id, embedding.as_slice())?;
        Ok(())
    }

    #[cfg(test)]
    fn insert_legacy_record_without_index(&self, summary: &str, created_at: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO memories (summary, created_at, embedding_json) VALUES (?1, ?2, NULL)",
            params![summary, created_at],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    #[cfg(test)]
    fn set_record_embedding_json(&self, id: i64, embedding_json: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE memories SET embedding_json = ?1 WHERE id = ?2",
            params![embedding_json, id],
        )?;
        Ok(())
    }
}

fn rank_memory(
    mode: MemorySearchMode,
    hybrid_lexical_weight: f64,
    hybrid_semantic_weight: f64,
    lexical_score: f64,
    semantic_score: f64,
    recency_score: f64,
) -> f64 {
    match mode {
        MemorySearchMode::Lexical => {
            if lexical_score <= 0.0 {
                return 0.0;
            }
            lexical_score + recency_score
        }
        MemorySearchMode::Semantic => {
            if semantic_score < 0.18 {
                return 0.0;
            }
            semantic_score * 10.0 + recency_score
        }
        MemorySearchMode::Hybrid => {
            if lexical_score <= 0.0 && semantic_score < 0.22 {
                return 0.0;
            }
            let weight_sum = (hybrid_lexical_weight + hybrid_semantic_weight).max(0.0001);
            let lexical_weight = hybrid_lexical_weight / weight_sum;
            let semantic_weight = hybrid_semantic_weight / weight_sum;
            (lexical_score * lexical_weight)
                + (semantic_score * 10.0 * semantic_weight)
                + recency_score
        }
    }
}

fn lexical_relevance(query: &str, summary: &str) -> f64 {
    let q_lower = query.to_lowercase();
    let s_lower = summary.to_lowercase();

    let q_tokens = tokenize(q_lower.as_str());
    let s_tokens = tokenize(s_lower.as_str());

    if q_tokens.is_empty() || s_tokens.is_empty() {
        if s_lower.contains(q_lower.as_str()) {
            return 1.0;
        }
        return 0.0;
    }

    let overlap = q_tokens
        .iter()
        .filter(|token| s_tokens.contains(*token))
        .count() as f64;
    if overlap == 0.0 && !s_lower.contains(q_lower.as_str()) {
        return 0.0;
    }

    let phrase_bonus = if s_lower.contains(q_lower.as_str()) {
        3.0
    } else {
        0.0
    };

    overlap * 5.0 + phrase_bonus
}

fn recency_boost(created_at: &str) -> f64 {
    let recency_days = parse_age_days(created_at).unwrap_or(365.0);
    2.0 / (1.0 + recency_days.max(0.0))
}

fn parse_age_days(created_at: &str) -> Option<f64> {
    let created = DateTime::parse_from_rfc3339(created_at).ok()?;
    let age = Utc::now().signed_duration_since(created.with_timezone(&Utc));
    Some(age.num_seconds().max(0) as f64 / 86_400.0)
}

fn tokenize(input: &str) -> Vec<String> {
    input
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn semantic_similarity(query_embedding: &[f64], summary_embedding: &[f64]) -> f64 {
    if query_embedding.is_empty()
        || summary_embedding.is_empty()
        || query_embedding.len() != summary_embedding.len()
    {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut q_norm_sq = 0.0;
    let mut s_norm_sq = 0.0;
    for (q, s) in query_embedding.iter().zip(summary_embedding.iter()) {
        dot += q * s;
        q_norm_sq += q * q;
        s_norm_sq += s * s;
    }
    if q_norm_sq <= 0.0 || s_norm_sq <= 0.0 {
        return 0.0;
    }
    let cosine = dot / (q_norm_sq.sqrt() * s_norm_sq.sqrt());
    cosine.clamp(0.0, 1.0)
}

impl OpenAIEmbeddingClient {
    fn embed(&self, input: &str) -> Result<Vec<f64>> {
        #[derive(Deserialize)]
        struct EmbeddingResponse {
            data: Vec<EmbeddingData>,
        }
        #[derive(Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f64>,
        }

        let payload = serde_json::json!({
            "model": self.model,
            "input": input,
        });

        let url = format!("{}/embeddings", self.base_url);
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            anyhow::bail!("embedding provider returned {}", response.status());
        }

        let parsed: EmbeddingResponse = response.json()?;
        let embedding = parsed
            .data
            .first()
            .map(|d| d.embedding.clone())
            .unwrap_or_default();
        Ok(embedding)
    }
}

fn embed_text(input: &str, dimensions: usize) -> Vec<f64> {
    let dim = dimensions.max(16);
    let mut vec = vec![0.0; dim];
    let normalized = input.to_ascii_lowercase();
    let grams = trigram_tokens(normalized.as_str());
    if grams.is_empty() {
        return vec;
    }

    for gram in grams {
        let mut hasher = DefaultHasher::new();
        gram.hash(&mut hasher);
        let hash = hasher.finish();
        let idx = (hash as usize) % dim;
        let sign = if ((hash >> 7) & 1) == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign;
    }
    vec
}

fn vector_to_json(vec: &[f64]) -> String {
    serde_json::to_string(vec).unwrap_or_else(|_| "[]".to_string())
}

fn parse_vector_json(raw: &str) -> Option<Vec<f64>> {
    serde_json::from_str(raw).ok()
}

fn ann_bucket_keys(embedding: &[f64], probe_count: usize) -> Vec<String> {
    if embedding.is_empty() {
        return Vec::new();
    }
    let probes = probe_count.max(1);
    let mut out = Vec::with_capacity(probes);
    for probe in 0..probes {
        let mut hasher = DefaultHasher::new();
        probe.hash(&mut hasher);
        let step = (embedding.len() / 16).max(1);
        for (bit_idx, idx) in (probe % step..embedding.len())
            .step_by(step)
            .take(64)
            .enumerate()
        {
            let bit = embedding[idx].is_sign_positive();
            if bit {
                (bit_idx as u64).hash(&mut hasher);
            }
        }
        out.push(format!("p{}-{:016x}", probe, hasher.finish()));
    }
    out
}

fn trigram_tokens(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in tokenize(input) {
        if token.len() < 3 {
            out.push(token);
            continue;
        }
        let chars: Vec<char> = token.chars().collect();
        for i in 0..=(chars.len() - 3) {
            out.push(chars[i..i + 3].iter().collect());
        }
    }
    out
}

fn histogram_bucket_index(bounds: &[u64], value: u64) -> usize {
    for (idx, bound) in bounds.iter().enumerate() {
        if value <= *bound {
            return idx;
        }
    }
    bounds.len()
}

fn summarize_messages(messages: &[Message]) -> String {
    let mut chunks = Vec::new();
    for msg in messages.iter().rev().take(6).rev() {
        let who = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
            Role::Tool => "tool",
        };
        chunks.push(format!("{who}: {}", msg.content.replace('\n', " ")));
    }
    let mut summary = chunks.join(" | ");
    if summary.len() > 500 {
        summary.truncate(500);
    }
    summary
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use anyhow::Result;
    use chrono::{Duration, Utc};

    use super::{
        MemoryWorker, ann_bucket_keys, embed_text, histogram_bucket_index, parse_vector_json,
        semantic_similarity, vector_to_json,
    };
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

    fn make_history(count: usize) -> Vec<Message> {
        (0..count)
            .map(|idx| Message::new(Role::User, format!("m-{idx}")))
            .collect()
    }

    #[test]
    fn maybe_snapshot_writes_summary_after_window() -> Result<()> {
        let dir = temp_test_dir("memory_snapshot");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;

        let none = worker.maybe_snapshot(&make_history(4))?;
        assert!(none.is_none());

        let created = worker.maybe_snapshot(&make_history(8))?;
        assert!(created.is_some());

        let latest = worker.latest(5)?;
        assert_eq!(latest.len(), 1);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn search_returns_ranked_matches() -> Result<()> {
        let dir = temp_test_dir("memory_search_ranked");
        let worker = MemoryWorker::open(dir.join("memory.db"))?;

        let now = Utc::now();
        worker.insert_test_record(
            "user: rust sqlite memory ranking",
            &(now - Duration::hours(1)).to_rfc3339(),
        )?;
        worker.insert_test_record("user: rust memory", &(now - Duration::days(2)).to_rfc3339())?;
        worker.insert_test_record(
            "assistant: unrelated networking topic",
            &(now - Duration::minutes(10)).to_rfc3339(),
        )?;

        let rows = worker.search("rust memory ranking", 5)?;
        assert_eq!(rows.len(), 2);
        assert!(rows[0].summary.contains("ranking"));
        assert!(rows[0].score >= rows[1].score);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn semantic_search_matches_morphological_similarity_without_exact_token_overlap() -> Result<()>
    {
        let dir = temp_test_dir("memory_search_semantic");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_search_profile("semantic", 0.0, 1.0);

        let now = Utc::now();
        worker.insert_test_record(
            "assistant: authentication rotation policy with grace window",
            &(now - Duration::minutes(5)).to_rfc3339(),
        )?;
        worker.insert_test_record(
            "assistant: random unrelated graphics rendering",
            &(now - Duration::minutes(1)).to_rfc3339(),
        )?;

        let rows = worker.search("auth rotate policy grace", 5)?;
        assert!(!rows.is_empty());
        assert!(rows[0].summary.contains("rotation policy"));

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn embedding_similarity_prefers_related_text() {
        let a = embed_text("auth token rotation grace window", 128);
        let b = embed_text("authentication token rotate with grace", 128);
        let c = embed_text("frontend css animation and typography", 128);

        let ab = semantic_similarity(a.as_slice(), b.as_slice());
        let ac = semantic_similarity(a.as_slice(), c.as_slice());
        assert!(ab > ac);
    }

    #[test]
    fn vector_json_roundtrip_is_lossless_enough_for_search() {
        let vec = vec![0.1, -0.25, 0.33, 1.0];
        let encoded = vector_to_json(vec.as_slice());
        let decoded = parse_vector_json(encoded.as_str()).expect("vector should parse");
        assert_eq!(decoded.len(), vec.len());
        assert!((decoded[0] - vec[0]).abs() < 1e-10);
        assert!((decoded[1] - vec[1]).abs() < 1e-10);
    }

    #[test]
    fn openai_provider_without_key_falls_back_to_local() -> Result<()> {
        let dir = temp_test_dir("memory_embed_provider_fallback");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_embedding_backend("openai", "", "", "MISSING_API_KEY_ENV", 128);
        worker.set_search_profile("semantic", 0.0, 1.0);

        let now = Utc::now();
        worker.insert_test_record(
            "assistant: token rotation with grace period",
            &(now - Duration::minutes(2)).to_rfc3339(),
        )?;
        let rows = worker.search("rotate token grace", 3)?;
        assert!(!rows.is_empty());

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn ann_bucket_keys_are_generated_for_non_empty_embeddings() {
        let emb = embed_text("routing ownership acl websocket session", 128);
        let keys = ann_bucket_keys(emb.as_slice(), 4);
        assert_eq!(keys.len(), 4);
        assert!(keys.iter().all(|k| k.starts_with('p')));
    }

    #[test]
    fn ann_candidate_query_returns_indexed_rows() -> Result<()> {
        let dir = temp_test_dir("memory_ann_candidates");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_ann_profile(true, 1, 6, 128);
        worker.set_search_profile("semantic", 0.0, 1.0);

        let now = Utc::now();
        worker.insert_test_record(
            "assistant: websocket session owner routing acl rules",
            &(now - Duration::minutes(3)).to_rfc3339(),
        )?;
        worker.insert_test_record(
            "assistant: unrelated graphics texture pipeline",
            &(now - Duration::minutes(2)).to_rfc3339(),
        )?;

        let q = worker.embed("websocket session acl owner");
        let keys = ann_bucket_keys(q.as_slice(), 6);
        let rows = worker.fetch_ann_candidates(keys.as_slice(), 64)?;
        assert!(!rows.is_empty());

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn maintenance_backfills_legacy_rows_missing_index() -> Result<()> {
        let dir = temp_test_dir("memory_ann_maintenance_backfill");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_ann_profile(true, 1, 6, 128);

        let now = Utc::now();
        let _id = worker.insert_legacy_record_without_index(
            "assistant: legacy row without ann index",
            &(now - Duration::minutes(6)).to_rfc3339(),
        )?;

        let stats = worker.maintain_ann_index(16)?;
        assert!(stats.backfilled_rows >= 1);

        let rows = worker.search("legacy ann index", 5)?;
        assert!(!rows.is_empty());

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn maintenance_reindexes_rows_with_drifted_embedding_shape() -> Result<()> {
        let dir = temp_test_dir("memory_ann_maintenance_reindex");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_ann_profile(true, 1, 6, 128);

        let now = Utc::now();
        worker.insert_test_record(
            "assistant: websocket auth token rotation and grace",
            &(now - Duration::minutes(4)).to_rfc3339(),
        )?;
        let latest = worker.latest(1)?;
        let id = latest[0].id;
        worker.set_record_embedding_json(id, "[0.1,0.2,0.3]")?;

        let stats = worker.maintain_ann_index(16)?;
        assert!(stats.reindexed_rows >= 1);

        let rows = worker.search("token rotation grace", 5)?;
        assert!(!rows.is_empty());

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn compact_removes_ttl_and_overflow_rows() -> Result<()> {
        let dir = temp_test_dir("memory_compact");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_retention(7, 2);

        let now = Utc::now();
        worker.insert_test_record("very old", &(now - Duration::days(20)).to_rfc3339())?;
        worker.insert_test_record("new-1", &(now - Duration::days(1)).to_rfc3339())?;
        worker.insert_test_record("new-2", &(now - Duration::hours(10)).to_rfc3339())?;
        worker.insert_test_record("new-3", &(now - Duration::hours(1)).to_rfc3339())?;

        let stats = worker.compact()?;
        assert_eq!(stats.removed_ttl, 1);
        assert_eq!(stats.remaining, 2);
        assert_eq!(worker.latest(10)?.len(), 2);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn diagnostics_track_ann_hit_and_fallback_rates() -> Result<()> {
        let dir = temp_test_dir("memory_diagnostics_rates");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_ann_profile(true, 32, 6, 256);
        worker.set_search_profile("semantic", 0.0, 1.0);

        let now = Utc::now();
        for idx in 0..40 {
            worker.insert_test_record(
                format!("assistant: websocket routing acl ownership sample-{idx}").as_str(),
                &(now - Duration::minutes(idx)).to_rfc3339(),
            )?;
        }

        let _ = worker.search("websocket acl routing ownership", 5)?;
        worker.conn.execute("DELETE FROM memory_ann_index", [])?;
        let _ = worker.search("websocket acl routing ownership", 5)?;

        let stats = worker.diagnostics()?;
        assert_eq!(stats.search_total, 2);
        assert_eq!(stats.ann_attempt_total, 2);
        assert_eq!(stats.ann_hit_total, 1);
        assert_eq!(stats.ann_fallback_total, 1);
        assert_eq!(stats.scan_only_total, 0);
        assert!((stats.ann_hit_rate - 0.5).abs() < 1e-9);
        assert!((stats.ann_fallback_rate - 0.5).abs() < 1e-9);
        assert!(stats.candidate_examined_total >= 2);
        assert!(stats.candidate_examined_avg >= 1.0);
        assert!(stats.candidate_examined_p95 >= 1);
        assert!(stats.query_latency_avg_ms >= 0.0);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn diagnostics_report_maintenance_lag_and_timestamp() -> Result<()> {
        let dir = temp_test_dir("memory_diagnostics_maintenance");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_ann_profile(true, 32, 6, 128);

        let now = Utc::now();
        worker.insert_legacy_record_without_index(
            "assistant: legacy maintenance diagnostics row",
            &(now - Duration::minutes(5)).to_rfc3339(),
        )?;

        let before = worker.diagnostics()?;
        assert!(before.remaining_unindexed >= 1);
        assert!(before.ann_last_maintained_at.is_none());

        let _ = worker.maintain_ann_index(8)?;
        let after = worker.diagnostics()?;
        assert_eq!(after.remaining_unindexed, 0);
        assert!(after.ann_cursor_max >= after.ann_cursor);
        assert!(after.ann_cursor_progress >= 0.0);
        assert!(after.ann_cursor_progress <= 1.0);
        assert!(after.ann_last_maintained_at.is_some());

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn diagnostics_track_scan_only_path_when_ann_is_not_used() -> Result<()> {
        let dir = temp_test_dir("memory_diagnostics_scan_only");
        let mut worker = MemoryWorker::open(dir.join("memory.db"))?;
        worker.set_search_profile("lexical", 1.0, 0.0);

        let now = Utc::now();
        worker.insert_test_record(
            "assistant: lexical scan path diagnostics",
            &(now - Duration::minutes(1)).to_rfc3339(),
        )?;
        let _ = worker.search("lexical diagnostics", 5)?;

        let stats = worker.diagnostics()?;
        assert_eq!(stats.search_total, 1);
        assert_eq!(stats.ann_attempt_total, 0);
        assert_eq!(stats.scan_only_total, 1);
        assert!((stats.scan_only_rate - 1.0).abs() < 1e-9);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn histogram_bucket_index_assigns_expected_boundaries() {
        assert_eq!(histogram_bucket_index(&[8, 16, 32], 0), 0);
        assert_eq!(histogram_bucket_index(&[8, 16, 32], 8), 0);
        assert_eq!(histogram_bucket_index(&[8, 16, 32], 9), 1);
        assert_eq!(histogram_bucket_index(&[8, 16, 32], 17), 2);
        assert_eq!(histogram_bucket_index(&[8, 16, 32], 40), 3);
    }
}
