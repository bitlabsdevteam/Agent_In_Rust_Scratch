use std::collections::HashMap;
use std::net::TcpListener;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tungstenite::Message;

#[derive(Debug, Clone)]
pub struct WebSocketInbound {
    pub session_id: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct WebSocketAuthConfig {
    pub required: bool,
    pub token: Option<String>,
    pub previous_token: Option<String>,
}

impl WebSocketAuthConfig {
    pub fn disabled() -> Self {
        Self {
            required: false,
            token: None,
            previous_token: None,
        }
    }

    pub fn token_required(token: impl Into<String>) -> Self {
        Self {
            required: true,
            token: Some(token.into()),
            previous_token: None,
        }
    }

    pub fn token_required_with_previous(
        token: impl Into<String>,
        previous_token: Option<String>,
    ) -> Self {
        Self {
            required: true,
            token: Some(token.into()),
            previous_token,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WebSocketRuntimeConfig {
    pub auth: WebSocketAuthConfig,
    pub max_clients: usize,
    pub idle_timeout_secs: u64,
    pub ping_interval_secs: u64,
    pub auth_rotation_grace_secs: u64,
}

impl Default for WebSocketRuntimeConfig {
    fn default() -> Self {
        Self {
            auth: WebSocketAuthConfig::disabled(),
            max_clients: 128,
            idle_timeout_secs: 300,
            ping_interval_secs: 30,
            auth_rotation_grace_secs: 300,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WebSocketPolicySnapshot {
    pub auth_required: bool,
    pub has_current_token: bool,
    pub has_previous_token: bool,
    pub max_clients: usize,
    pub idle_timeout_secs: u64,
    pub ping_interval_secs: u64,
    pub auth_rotation_grace_secs: u64,
}

#[derive(Debug, Clone)]
struct WebSocketRuntimePolicy {
    auth_required: bool,
    current_token: Option<String>,
    previous_token: Option<String>,
    previous_token_valid_until: Option<Instant>,
    max_clients: usize,
    idle_timeout: Duration,
    ping_interval: Duration,
    auth_rotation_grace: Duration,
}

impl WebSocketRuntimePolicy {
    fn from_config(cfg: &WebSocketRuntimeConfig) -> Self {
        let now = Instant::now();
        let previous_token = cfg.auth.previous_token.clone();
        let previous_token_valid_until = if previous_token.is_some() {
            Some(now + Duration::from_secs(cfg.auth_rotation_grace_secs))
        } else {
            None
        };

        Self {
            auth_required: cfg.auth.required,
            current_token: cfg.auth.token.clone(),
            previous_token,
            previous_token_valid_until,
            max_clients: cfg.max_clients.max(1),
            idle_timeout: Duration::from_secs(cfg.idle_timeout_secs),
            ping_interval: Duration::from_secs(cfg.ping_interval_secs),
            auth_rotation_grace: Duration::from_secs(cfg.auth_rotation_grace_secs),
        }
    }

    fn apply_config(&mut self, cfg: &WebSocketRuntimeConfig) {
        let previous_changed = self.previous_token != cfg.auth.previous_token;
        self.auth_required = cfg.auth.required;
        self.current_token = cfg.auth.token.clone();
        self.max_clients = cfg.max_clients.max(1);
        self.idle_timeout = Duration::from_secs(cfg.idle_timeout_secs);
        self.ping_interval = Duration::from_secs(cfg.ping_interval_secs);
        self.auth_rotation_grace = Duration::from_secs(cfg.auth_rotation_grace_secs);

        if previous_changed {
            self.previous_token = cfg.auth.previous_token.clone();
            self.previous_token_valid_until = if self.previous_token.is_some() {
                Some(Instant::now() + self.auth_rotation_grace)
            } else {
                None
            };
        } else if self.previous_token.is_none() {
            self.previous_token_valid_until = None;
        }
    }

    fn rotate_auth_token(&mut self, new_token: &str) {
        let candidate = new_token.trim();
        if candidate.is_empty() {
            return;
        }
        if self.current_token.as_deref() == Some(candidate) {
            return;
        }
        self.previous_token = self.current_token.take();
        self.previous_token_valid_until = if self.previous_token.is_some() {
            Some(Instant::now() + self.auth_rotation_grace)
        } else {
            None
        };
        self.current_token = Some(candidate.to_string());
        self.auth_required = true;
    }

    fn auth_valid(&self, message: &str) -> bool {
        if !self.auth_required {
            return true;
        }

        let Some(provided) = parse_auth_token(message) else {
            return false;
        };

        if self.current_token.as_deref() == Some(provided) {
            return true;
        }

        if self.previous_token.as_deref() == Some(provided) {
            if let Some(deadline) = self.previous_token_valid_until {
                return Instant::now() <= deadline;
            }
        }
        false
    }

    fn is_auth_required(&self) -> bool {
        self.auth_required
    }

    fn snapshot(&self) -> WebSocketPolicySnapshot {
        WebSocketPolicySnapshot {
            auth_required: self.auth_required,
            has_current_token: self.current_token.is_some(),
            has_previous_token: self.previous_token.is_some(),
            max_clients: self.max_clients,
            idle_timeout_secs: self.idle_timeout.as_secs(),
            ping_interval_secs: self.ping_interval.as_secs(),
            auth_rotation_grace_secs: self.auth_rotation_grace.as_secs(),
        }
    }
}

pub struct WebSocketServer {
    enabled: bool,
    clients: Arc<Mutex<HashMap<String, Sender<String>>>>,
    inbound_rx: Receiver<WebSocketInbound>,
    policy: Arc<RwLock<WebSocketRuntimePolicy>>,
    stop_tx: Option<Sender<()>>,
    accept_handle: Option<JoinHandle<()>>,
}

impl WebSocketServer {
    pub fn start(
        bind_addr: &str,
        enabled: bool,
        runtime_cfg: WebSocketRuntimeConfig,
    ) -> Result<Self> {
        let (inbound_tx, inbound_rx) = mpsc::channel::<WebSocketInbound>();
        let clients = Arc::new(Mutex::new(HashMap::<String, Sender<String>>::new()));
        let policy = Arc::new(RwLock::new(WebSocketRuntimePolicy::from_config(
            &runtime_cfg,
        )));

        if !enabled {
            return Ok(Self {
                enabled: false,
                clients,
                inbound_rx,
                policy,
                stop_tx: None,
                accept_handle: None,
            });
        }

        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("failed to bind websocket listener at {bind_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("failed to set websocket listener non-blocking")?;

        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let clients_clone = Arc::clone(&clients);
        let policy_clone = Arc::clone(&policy);
        let accept_handle = thread::spawn(move || {
            let next_id = AtomicU64::new(1);
            loop {
                if stop_rx.try_recv().is_ok() {
                    break;
                }

                let accepted = listener.accept();
                let (stream, addr) = match accepted {
                    Ok(v) => v,
                    Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                    Err(_) => {
                        thread::sleep(Duration::from_millis(100));
                        continue;
                    }
                };

                let over_quota = {
                    let max_clients = policy_clone.read().map(|p| p.max_clients).unwrap_or(1);
                    let active_clients = clients_clone.lock().map(|c| c.len()).unwrap_or(0);
                    !can_accept_client(active_clients, max_clients)
                };
                if over_quota {
                    continue;
                }

                let ws = match tungstenite::accept(stream) {
                    Ok(ws) => ws,
                    Err(_) => continue,
                };

                let session_id = format!("ws-{}-{}", next_id.fetch_add(1, Ordering::Relaxed), addr);
                let (out_tx, out_rx) = mpsc::channel::<String>();
                if let Ok(mut locked) = clients_clone.lock() {
                    locked.insert(session_id.clone(), out_tx);
                }

                let inbound_tx_conn = inbound_tx.clone();
                let clients_remove = Arc::clone(&clients_clone);
                let policy_conn = Arc::clone(&policy_clone);
                thread::spawn(move || {
                    run_client_loop(
                        ws,
                        session_id,
                        inbound_tx_conn,
                        out_rx,
                        clients_remove,
                        policy_conn,
                    )
                });
            }
        });

        Ok(Self {
            enabled: true,
            clients,
            inbound_rx,
            policy,
            stop_tx: Some(stop_tx),
            accept_handle: Some(accept_handle),
        })
    }

    pub fn drain_inbound(&self) -> Vec<WebSocketInbound> {
        self.inbound_rx.try_iter().collect()
    }

    pub fn send(&self, session_id: Option<&str>, content: &str) {
        if !self.enabled {
            return;
        }
        let msg = content.to_string();
        if let Ok(locked) = self.clients.lock() {
            if let Some(id) = session_id {
                if let Some(tx) = locked.get(id) {
                    let _ = tx.send(msg);
                }
                return;
            }

            for tx in locked.values() {
                let _ = tx.send(msg.clone());
            }
        }
    }

    pub fn client_count(&self) -> usize {
        if let Ok(locked) = self.clients.lock() {
            return locked.len();
        }
        0
    }

    pub fn reconfigure(&self, runtime_cfg: WebSocketRuntimeConfig) {
        if let Ok(mut policy) = self.policy.write() {
            policy.apply_config(&runtime_cfg);
        }
    }

    pub fn rotate_auth_token(&self, new_token: &str) {
        if let Ok(mut policy) = self.policy.write() {
            policy.rotate_auth_token(new_token);
        }
    }

    pub fn policy_snapshot(&self) -> WebSocketPolicySnapshot {
        if let Ok(policy) = self.policy.read() {
            return policy.snapshot();
        }
        WebSocketPolicySnapshot {
            auth_required: false,
            has_current_token: false,
            has_previous_token: false,
            max_clients: 1,
            idle_timeout_secs: 0,
            ping_interval_secs: 0,
            auth_rotation_grace_secs: 0,
        }
    }

    pub fn stop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.accept_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for WebSocketServer {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run_client_loop(
    mut ws: tungstenite::WebSocket<std::net::TcpStream>,
    session_id: String,
    inbound_tx: Sender<WebSocketInbound>,
    out_rx: Receiver<String>,
    clients: Arc<Mutex<HashMap<String, Sender<String>>>>,
    policy: Arc<RwLock<WebSocketRuntimePolicy>>,
) {
    let _ = ws
        .get_mut()
        .set_read_timeout(Some(Duration::from_millis(100)));

    let mut authenticated = !is_auth_required(&policy);
    if is_auth_required(&policy) {
        let _ = ws.send(Message::Text(
            "AUTH_REQUIRED send: AUTH <token>".to_string(),
        ));
    } else {
        let _ = ws.send(Message::Text("READY".to_string()));
    }

    let mut last_activity = Instant::now();
    let mut last_ping = Instant::now();

    loop {
        let (idle_timeout, ping_interval) = read_timing_config(&policy);

        if ping_interval > Duration::ZERO && last_ping.elapsed() >= ping_interval {
            if ws.send(Message::Ping(Vec::new())).is_err() {
                cleanup_client(&session_id, &clients);
                return;
            }
            last_ping = Instant::now();
        }

        if idle_timeout > Duration::ZERO && last_activity.elapsed() > idle_timeout {
            let _ = ws.send(Message::Close(None));
            cleanup_client(&session_id, &clients);
            return;
        }

        loop {
            match out_rx.try_recv() {
                Ok(payload) => {
                    if ws.send(Message::Text(payload)).is_err() {
                        cleanup_client(&session_id, &clients);
                        return;
                    }
                    last_activity = Instant::now();
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    cleanup_client(&session_id, &clients);
                    return;
                }
            }
        }

        match ws.read() {
            Ok(Message::Text(text)) => {
                last_activity = Instant::now();
                if !authenticated {
                    if is_auth_valid(text.as_str(), &policy) {
                        authenticated = true;
                        let _ = ws.send(Message::Text("AUTH_OK".to_string()));
                    } else {
                        let _ = ws.send(Message::Text("AUTH_FAIL".to_string()));
                    }
                    continue;
                }

                let _ = inbound_tx.send(WebSocketInbound {
                    session_id: session_id.clone(),
                    content: text.to_string(),
                });
            }
            Ok(Message::Binary(_)) => {
                last_activity = Instant::now();
            }
            Ok(Message::Ping(payload)) => {
                last_activity = Instant::now();
                if ws.send(Message::Pong(payload)).is_err() {
                    cleanup_client(&session_id, &clients);
                    return;
                }
            }
            Ok(Message::Pong(_)) => {
                last_activity = Instant::now();
            }
            Ok(Message::Close(_)) => {
                cleanup_client(&session_id, &clients);
                return;
            }
            Ok(Message::Frame(_)) => {}
            Err(tungstenite::Error::Io(err))
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => {
                cleanup_client(&session_id, &clients);
                return;
            }
        }
    }
}

fn parse_auth_token(message: &str) -> Option<&str> {
    let trimmed = message.trim();
    let provided = trimmed.strip_prefix("AUTH ")?;
    Some(provided.trim())
}

fn is_auth_valid(message: &str, policy: &Arc<RwLock<WebSocketRuntimePolicy>>) -> bool {
    if let Ok(policy) = policy.read() {
        return policy.auth_valid(message);
    }
    false
}

fn is_auth_required(policy: &Arc<RwLock<WebSocketRuntimePolicy>>) -> bool {
    if let Ok(policy) = policy.read() {
        return policy.is_auth_required();
    }
    false
}

fn read_timing_config(policy: &Arc<RwLock<WebSocketRuntimePolicy>>) -> (Duration, Duration) {
    if let Ok(policy) = policy.read() {
        return (policy.idle_timeout, policy.ping_interval);
    }
    (Duration::ZERO, Duration::ZERO)
}

fn can_accept_client(active_clients: usize, max_clients: usize) -> bool {
    active_clients < max_clients.max(1)
}

fn cleanup_client(session_id: &str, clients: &Arc<Mutex<HashMap<String, Sender<String>>>>) {
    if let Ok(mut locked) = clients.lock() {
        locked.remove(session_id);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use super::{
        WebSocketAuthConfig, WebSocketRuntimeConfig, WebSocketRuntimePolicy, WebSocketServer,
        can_accept_client, is_auth_valid,
    };

    #[test]
    fn disabled_server_is_noop() {
        let runtime = WebSocketRuntimeConfig {
            auth: WebSocketAuthConfig::disabled(),
            ..WebSocketRuntimeConfig::default()
        };
        let server = WebSocketServer::start("127.0.0.1:0", false, runtime)
            .expect("disabled websocket server should initialize");
        assert_eq!(server.client_count(), 0);
        assert!(server.drain_inbound().is_empty());
    }

    #[test]
    fn auth_message_validation_uses_current_and_rotated_token_rules() {
        let runtime = WebSocketRuntimeConfig {
            auth: WebSocketAuthConfig::token_required("old"),
            auth_rotation_grace_secs: 0,
            ..WebSocketRuntimeConfig::default()
        };
        let mut policy = WebSocketRuntimePolicy::from_config(&runtime);
        policy.rotate_auth_token("new");
        let policy = Arc::new(RwLock::new(policy));

        assert!(is_auth_valid("AUTH new", &policy));
        assert!(!is_auth_valid("AUTH old", &policy));
        assert!(!is_auth_valid("new", &policy));
    }

    #[test]
    fn auth_config_builders_work() {
        let disabled = WebSocketAuthConfig::disabled();
        assert!(!disabled.required);
        assert!(disabled.token.is_none());

        let required = WebSocketAuthConfig::token_required("abc");
        assert!(required.required);
        assert_eq!(required.token.as_deref(), Some("abc"));
        assert!(required.previous_token.is_none());
    }

    #[test]
    fn max_client_quota_check_is_strict() {
        assert!(can_accept_client(0, 1));
        assert!(!can_accept_client(1, 1));
        assert!(can_accept_client(2, 3));
    }
}
