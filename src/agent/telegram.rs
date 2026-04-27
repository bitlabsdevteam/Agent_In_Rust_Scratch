use std::collections::HashSet;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct TelegramInbound {
    pub chat_id: i64,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramRuntimeConfig {
    pub enabled: bool,
    pub bot_token: String,
    pub api_base_url: String,
    pub poll_interval_secs: u64,
    pub poll_timeout_secs: u64,
    pub allowed_chat_ids: Vec<i64>,
}

impl Default for TelegramRuntimeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            bot_token: String::new(),
            api_base_url: "https://api.telegram.org".to_string(),
            poll_interval_secs: 2,
            poll_timeout_secs: 0,
            allowed_chat_ids: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TelegramPolicySnapshot {
    pub enabled: bool,
    pub has_bot_token: bool,
    pub api_base_url: String,
    pub poll_interval_secs: u64,
    pub poll_timeout_secs: u64,
    pub allowed_chat_ids_count: usize,
}

pub struct TelegramRuntime {
    client: Client,
    cfg: TelegramRuntimeConfig,
    allowed_chat_ids: HashSet<i64>,
    last_update_id: i64,
    next_poll_at: Instant,
}

impl TelegramRuntime {
    pub fn start(cfg: TelegramRuntimeConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .context("failed to build telegram http client")?;
        let allowed_chat_ids = cfg.allowed_chat_ids.iter().copied().collect();
        Ok(Self {
            client,
            cfg,
            allowed_chat_ids,
            last_update_id: 0,
            next_poll_at: Instant::now(),
        })
    }

    pub fn reconfigure(&mut self, cfg: TelegramRuntimeConfig) {
        self.allowed_chat_ids = cfg.allowed_chat_ids.iter().copied().collect();
        self.cfg = cfg;
        self.next_poll_at = Instant::now();
    }

    pub fn policy_snapshot(&self) -> TelegramPolicySnapshot {
        TelegramPolicySnapshot {
            enabled: self.cfg.enabled,
            has_bot_token: !self.cfg.bot_token.trim().is_empty(),
            api_base_url: self.cfg.api_base_url.clone(),
            poll_interval_secs: self.cfg.poll_interval_secs,
            poll_timeout_secs: self.cfg.poll_timeout_secs,
            allowed_chat_ids_count: self.allowed_chat_ids.len(),
        }
    }

    pub fn drain_inbound(&mut self) -> Result<Vec<TelegramInbound>> {
        if !self.is_active() {
            return Ok(Vec::new());
        }
        if Instant::now() < self.next_poll_at {
            return Ok(Vec::new());
        }
        self.next_poll_at = Instant::now() + Duration::from_secs(self.cfg.poll_interval_secs);

        let mut out = Vec::new();
        for update in self.fetch_updates()? {
            self.last_update_id = self.last_update_id.max(update.update_id);
            let Some(message) = update.message else {
                continue;
            };
            let Some(text) = message.text else {
                continue;
            };
            let chat_id = message.chat.id;
            if !self.chat_allowed(chat_id) {
                continue;
            }
            out.push(TelegramInbound {
                chat_id,
                content: text,
            });
        }
        Ok(out)
    }

    pub fn send_message(&self, chat_id: i64, content: &str) -> Result<()> {
        if !self.is_active() {
            return Ok(());
        }
        if !self.chat_allowed(chat_id) {
            return Err(anyhow!(
                "telegram chat is not allowed by allowlist: {chat_id}"
            ));
        }

        let endpoint = format!(
            "{}/bot{}/sendMessage",
            self.cfg.api_base_url.trim_end_matches('/'),
            self.cfg.bot_token.trim()
        );
        let body = SendMessageRequest {
            chat_id,
            text: content,
        };
        let response = self
            .client
            .post(endpoint)
            .json(&body)
            .send()
            .context("telegram sendMessage request failed")?;
        let decoded: TelegramApiResponse<serde_json::Value> = response
            .json()
            .context("failed to decode telegram sendMessage response")?;
        if !decoded.ok {
            return Err(anyhow!(
                "telegram sendMessage rejected: {}",
                decoded
                    .description
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }
        Ok(())
    }

    fn is_active(&self) -> bool {
        self.cfg.enabled && !self.cfg.bot_token.trim().is_empty()
    }

    fn chat_allowed(&self, chat_id: i64) -> bool {
        self.allowed_chat_ids.is_empty() || self.allowed_chat_ids.contains(&chat_id)
    }

    fn fetch_updates(&self) -> Result<Vec<TelegramUpdate>> {
        let endpoint = format!(
            "{}/bot{}/getUpdates",
            self.cfg.api_base_url.trim_end_matches('/'),
            self.cfg.bot_token.trim()
        );
        let body = GetUpdatesRequest {
            offset: self.last_update_id + 1,
            timeout: self.cfg.poll_timeout_secs,
            allowed_updates: vec!["message"],
        };

        let response = self
            .client
            .post(endpoint)
            .json(&body)
            .send()
            .context("telegram getUpdates request failed")?;
        let decoded: TelegramApiResponse<Vec<TelegramUpdate>> = response
            .json()
            .context("failed to decode telegram getUpdates response")?;
        if !decoded.ok {
            return Err(anyhow!(
                "telegram getUpdates rejected: {}",
                decoded
                    .description
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }
        Ok(decoded.result.unwrap_or_default())
    }
}

#[derive(Debug, Serialize)]
struct GetUpdatesRequest<'a> {
    offset: i64,
    timeout: u64,
    allowed_updates: Vec<&'a str>,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest<'a> {
    chat_id: i64,
    text: &'a str,
}

#[derive(Debug, Deserialize)]
struct TelegramApiResponse<T> {
    ok: bool,
    result: Option<T>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    message: Option<TelegramMessage>,
}

#[derive(Debug, Deserialize)]
struct TelegramMessage {
    chat: TelegramChat,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramChat {
    id: i64,
}
