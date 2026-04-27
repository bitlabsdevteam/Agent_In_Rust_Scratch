use std::collections::HashMap;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{self, IsTerminal, Write};
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};

use crate::agent::config::{ConfigManager, HarnessConfig};
use crate::agent::events::{
    DeliveryTarget, DispatchTask, EventBus, EventSource, InboundEvent, InboundPayload,
    OutboundEvent,
};
use crate::agent::memory::MemoryWorker;
use crate::agent::model::{ModelBackend, ModelOutput, ModelResponse, ModelRuntimeEvent};
use crate::agent::permissions::PermissionPolicy;
use crate::agent::planner::PlannerEngine;
use crate::agent::policy::{PolicyContext, PolicyEngine};
use crate::agent::prompt::{PromptLayers, assemble_prompt};
use crate::agent::router::SourceRouter;
use crate::agent::scheduler::{SchedulerControl, SchedulerDaemon, enqueue_heartbeat};
use crate::agent::session::SessionStore;
use crate::agent::skills::{SkillRegistry, default_skill_dir};
use crate::agent::telegram::{TelegramRuntime, TelegramRuntimeConfig};
use crate::agent::tools::{ToolContext, ToolRegistry};
use crate::agent::types::{Message, Role};
use crate::agent::websocket::{WebSocketAuthConfig, WebSocketRuntimeConfig, WebSocketServer};

type AgentHandler<M> =
    Arc<dyn Fn(&mut Harness<M>, DeliveryTarget, DispatchTask) -> Result<()> + Send + Sync>;

pub struct Harness<M: ModelBackend> {
    pub model: M,
    pub tools: ToolRegistry,
    pub permissions: PermissionPolicy,
    pub session: SessionStore,
    pub tool_ctx: ToolContext,
    pub history: Vec<Message>,
    pub compact_threshold_messages: usize,
    pub max_inbound_events_per_cycle: usize,
    pub max_outbound_events_per_cycle: usize,
    event_bus: EventBus,
    router: SourceRouter,
    agent_handlers: HashMap<String, AgentHandler<M>>,
    file_outbox_path: PathBuf,
    websocket_outbox: VecDeque<String>,
    telegram_outbox: VecDeque<String>,
    config_manager: ConfigManager,
    planner: PlannerEngine,
    policy_engine: PolicyEngine,
    scheduler_control: SchedulerControl,
    _scheduler_daemon: SchedulerDaemon,
    websocket_server: WebSocketServer,
    telegram_runtime: TelegramRuntime,
    skills: SkillRegistry,
    memory_worker: MemoryWorker,
    memory_ann_backfill_batch_size: usize,
    memory_ann_maintenance_interval_ticks: u64,
    memory_status_snapshot_interval_ticks: u64,
    runtime_tick_count: u64,
    active_policy_context: ActivePolicyContext,
    cli_reasoning_overlay: Arc<Mutex<CliReasoningOverlay>>,
    model_busy: Arc<AtomicBool>,
    cancel_requested: Arc<AtomicBool>,
}

#[derive(Clone)]
struct ActivePolicyContext {
    source: EventSource,
    channel: DeliveryTarget,
    agent: String,
    websocket_session_id: Option<String>,
    telegram_chat_id: Option<i64>,
}

struct CliReasoningOverlay {
    enabled: bool,
    is_tty: bool,
    max_lines: usize,
    rendered_lines: usize,
    lines: VecDeque<String>,
    working_elapsed: Option<Duration>,
    working_tick: usize,
}

impl CliReasoningOverlay {
    fn new(enabled: bool, max_lines: usize) -> Self {
        Self {
            enabled,
            is_tty: io::stdout().is_terminal(),
            max_lines,
            rendered_lines: 0,
            lines: VecDeque::new(),
            working_elapsed: None,
            working_tick: 0,
        }
    }

    fn set_enabled(&mut self, enabled: bool) -> io::Result<()> {
        if !enabled {
            self.clear()?;
        }
        self.enabled = enabled;
        Ok(())
    }

    fn push(&mut self, message: impl Into<String>) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.lines.push_back(message.into());
        while self.lines.len() > self.max_lines {
            self.lines.pop_front();
        }
        if !self.is_tty {
            return Ok(());
        }
        self.render()
    }

    fn clear(&mut self) -> io::Result<()> {
        if !self.is_tty || self.rendered_lines == 0 {
            self.lines.clear();
            self.working_elapsed = None;
            self.working_tick = 0;
            self.rendered_lines = 0;
            return Ok(());
        }
        let mut stdout = io::stdout();
        write!(stdout, "\x1b[{}A", self.rendered_lines)?;
        for _ in 0..self.rendered_lines {
            write!(stdout, "\r\x1b[2K\n")?;
        }
        write!(stdout, "\x1b[{}A", self.rendered_lines)?;
        stdout.flush()?;
        self.lines.clear();
        self.working_elapsed = None;
        self.working_tick = 0;
        self.rendered_lines = 0;
        Ok(())
    }

    fn start_working(&mut self) -> io::Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.working_elapsed = Some(Duration::from_secs(0));
        self.working_tick = 0;
        if !self.is_tty {
            return Ok(());
        }
        self.render()
    }

    fn tick_working(&mut self, elapsed: Duration) -> io::Result<()> {
        if !self.enabled || self.working_elapsed.is_none() {
            return Ok(());
        }
        self.working_elapsed = Some(elapsed);
        self.working_tick = self.working_tick.saturating_add(1);
        if !self.is_tty {
            return Ok(());
        }
        self.render()
    }

    fn stop_working(&mut self) -> io::Result<()> {
        if self.working_elapsed.is_none() {
            return Ok(());
        }
        self.working_elapsed = None;
        self.working_tick = 0;
        if !self.is_tty {
            return Ok(());
        }
        self.render()
    }

    fn render(&mut self) -> io::Result<()> {
        let previous_lines = self.rendered_lines;
        let mut stdout = io::stdout();
        if previous_lines > 0 {
            write!(stdout, "\x1b[{}A", previous_lines)?;
            for _ in 0..previous_lines {
                write!(stdout, "\r\x1b[2K\n")?;
            }
            write!(stdout, "\x1b[{}A", previous_lines)?;
        }

        let total = self.lines.len();
        for (idx, line) in self.lines.iter().enumerate() {
            let color = reasoning_line_color_code(idx, total);
            write!(stdout, "\r\x1b[2K\x1b[{color}m• {line}\x1b[0m\n")?;
        }
        if let Some(elapsed) = self.working_elapsed {
            let line = format_working_line(elapsed, self.working_tick);
            write!(stdout, "\r\x1b[2K\x1b[97m• {line}\x1b[0m\n")?;
        }
        stdout.flush()?;
        self.rendered_lines = total + usize::from(self.working_elapsed.is_some());
        Ok(())
    }
}

impl<M: ModelBackend> Harness<M> {
    pub fn new(
        model: M,
        tools: ToolRegistry,
        permissions: PermissionPolicy,
        session: SessionStore,
        tool_ctx: ToolContext,
        compact_threshold_messages: usize,
    ) -> Result<Self> {
        let history = session.load()?;
        let agent_dir = PathBuf::from(&tool_ctx.workspace_root).join(".agent");
        let event_db_path = agent_dir.join("events.db");
        let file_outbox_path = agent_dir.join("outbox.jsonl");
        let event_bus = EventBus::open(&event_db_path)?;

        let config_path = agent_dir.join("config.json");
        let mut config_manager = ConfigManager::new(&config_path);
        let mut cfg = config_manager.load()?;
        let defaults = HarnessConfig::default();
        if cfg.compact_threshold_messages == defaults.compact_threshold_messages {
            cfg.compact_threshold_messages = compact_threshold_messages;
        }

        let policy_path = resolve_workspace_path(&tool_ctx.workspace_root, &cfg.policy_file);
        let policy_audit_path = agent_dir.join("policy_audit.jsonl");
        let policy_engine = PolicyEngine::new(policy_path, policy_audit_path)?;

        let skill_dir = default_skill_dir(&tool_ctx.workspace_root, &cfg.skill_dir);
        let skills = SkillRegistry::load_from_dir(skill_dir)?;

        let artifact_dir = resolve_workspace_path(&tool_ctx.workspace_root, &cfg.artifacts_dir);
        let planner = PlannerEngine::new(artifact_dir);

        let scheduler_control = SchedulerControl::new(
            cfg.scheduler_enabled,
            cfg.heartbeat_interval_secs,
            cfg.max_scheduler_pending_inbound,
        );
        let scheduler_daemon =
            SchedulerDaemon::start(event_db_path.clone(), scheduler_control.clone());
        let ws_runtime = websocket_runtime_config_from_harness(&cfg);
        let tg_runtime = telegram_runtime_config_from_harness(&cfg);

        let websocket_server = match WebSocketServer::start(
            cfg.websocket_bind_addr.as_str(),
            cfg.websocket_enabled,
            ws_runtime,
        ) {
            Ok(server) => server,
            Err(err) => {
                eprintln!(
                    "websocket disabled: failed to start at {} ({err})",
                    cfg.websocket_bind_addr
                );
                WebSocketServer::start("127.0.0.1:0", false, WebSocketRuntimeConfig::default())?
            }
        };
        let telegram_runtime = TelegramRuntime::start(tg_runtime)?;

        let mut memory_worker = MemoryWorker::open(agent_dir.join("memory.db"))?;
        memory_worker.set_retention(
            cfg.memory_retention_ttl_days,
            cfg.memory_retention_max_records,
        );
        memory_worker.set_search_profile(
            cfg.memory_search_mode.as_str(),
            cfg.memory_hybrid_lexical_weight,
            cfg.memory_hybrid_semantic_weight,
        );
        memory_worker.set_embedding_backend(
            cfg.memory_embedding_provider.as_str(),
            cfg.memory_embedding_model.as_str(),
            cfg.memory_embedding_base_url.as_str(),
            cfg.memory_embedding_api_key_env.as_str(),
            cfg.memory_embedding_dimensions,
        );
        memory_worker.set_ann_profile(
            cfg.memory_ann_enabled,
            cfg.memory_ann_scan_threshold,
            cfg.memory_ann_probe_count,
            cfg.memory_ann_candidate_cap,
        );

        let mut harness = Self {
            model,
            tools,
            permissions,
            session,
            tool_ctx,
            history,
            compact_threshold_messages: cfg.compact_threshold_messages,
            max_inbound_events_per_cycle: cfg.max_inbound_events_per_cycle,
            max_outbound_events_per_cycle: cfg.max_outbound_events_per_cycle,
            event_bus,
            router: SourceRouter::default(),
            agent_handlers: HashMap::new(),
            file_outbox_path,
            websocket_outbox: VecDeque::new(),
            telegram_outbox: VecDeque::new(),
            config_manager,
            planner,
            policy_engine,
            scheduler_control,
            _scheduler_daemon: scheduler_daemon,
            websocket_server,
            telegram_runtime,
            skills,
            memory_worker,
            memory_ann_backfill_batch_size: cfg.memory_ann_backfill_batch_size,
            memory_ann_maintenance_interval_ticks: cfg.memory_ann_maintenance_interval_ticks,
            memory_status_snapshot_interval_ticks: cfg.memory_status_snapshot_interval_ticks,
            runtime_tick_count: 0,
            active_policy_context: ActivePolicyContext {
                source: EventSource::Cli,
                channel: DeliveryTarget::Cli,
                agent: "default".to_string(),
                websocket_session_id: None,
                telegram_chat_id: None,
            },
            cli_reasoning_overlay: Arc::new(Mutex::new(CliReasoningOverlay::new(true, 5))),
            model_busy: Arc::new(AtomicBool::new(false)),
            cancel_requested: Arc::new(AtomicBool::new(false)),
        };
        harness.set_source_route(&EventSource::Cli, "default");
        harness.set_source_route(&EventSource::WebSocket, "default");
        harness.set_source_route(&EventSource::Telegram, "default");
        harness.set_source_route(&EventSource::Scheduler, "default");
        harness.register_builtin_agent_handlers();
        Ok(harness)
    }

    pub fn run(&mut self) -> Result<()> {
        println!("Agent harness ready. Type `/help` or `/exit`.");
        let (input_tx, input_rx) = mpsc::channel::<String>();
        let mut pending_cli_inputs: VecDeque<String> = VecDeque::new();
        let model_busy = Arc::clone(&self.model_busy);
        let cancel_requested = Arc::clone(&self.cancel_requested);
        thread::spawn(move || {
            loop {
                let mut input = String::new();
                match io::stdin().read_line(&mut input) {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = input.trim().to_string();
                        if model_busy.load(Ordering::Relaxed) && trimmed == "/cancel" {
                            cancel_requested.store(true, Ordering::Relaxed);
                            println!("cancel> interrupt requested");
                            continue;
                        }
                        if model_busy.load(Ordering::Relaxed) && !trimmed.is_empty() {
                            println!("queued> {trimmed}");
                        }
                        if input_tx.send(input).is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        loop {
            if !self.is_model_busy() {
                if let Some(queued) = pending_cli_inputs.pop_front() {
                    self.handle_runtime_input(queued.trim())?;
                    self.reload_config_if_changed()?;
                    continue;
                }
            }

            self.ingest_websocket_inbound_events()?;
            self.ingest_telegram_inbound_events()?;
            self.process_inbound_events()?;
            self.flush_outbound_events()?;
            self.run_memory_maintenance_tick()?;

            match input_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(raw) => {
                    let trimmed = raw.trim().to_string();
                    if self.is_model_busy() && !trimmed.is_empty() {
                        pending_cli_inputs.push_back(trimmed.clone());
                        self.overlay_clear()?;
                        println!("queued> {trimmed}");
                    } else {
                        self.handle_runtime_input(trimmed.as_str())?;
                    }
                    self.reload_config_if_changed()?;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    self.reload_config_if_changed()?;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        Ok(())
    }

    fn handle_runtime_input(&mut self, input: &str) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        if self.handle_slash(input)? {
            return Ok(());
        }

        self.overlay_clear()?;
        println!("{}", render_cli_chat_line("user", input));

        if let Some((name, args)) = parse_tool_call(input) {
            let output = self.execute_tool(name, args)?;
            self.push_message(Role::Tool, format!("{name} {args}"))?;
            let rendered = format!("tool:{name} -> {output}");
            self.emit_assistant_output(DeliveryTarget::Cli, rendered)?;
            self.flush_outbound_events()?;
            self.auto_compact_if_needed();
            return Ok(());
        }

        self.event_bus
            .publish_inbound(InboundEvent::user_message(EventSource::Cli, input))?;
        self.process_inbound_events()?;
        self.flush_outbound_events()?;
        self.auto_compact_if_needed();
        Ok(())
    }

    fn run_model_turn(&mut self, target: DeliveryTarget) -> Result<()> {
        struct BusyGuard {
            flag: Arc<AtomicBool>,
        }

        impl Drop for BusyGuard {
            fn drop(&mut self) {
                self.flag.store(false, Ordering::Relaxed);
            }
        }

        self.model_busy.store(true, Ordering::Relaxed);
        let _busy_guard = BusyGuard {
            flag: Arc::clone(&self.model_busy),
        };

        let max_tool_rounds = 8;
        let cli_reasoning = target == DeliveryTarget::Cli;

        for _ in 0..max_tool_rounds {
            if cli_reasoning {
                self.overlay_push("I'm now assembling prompt context, tools, and memory.")?;
            }
            let specs = self.tools.specs();
            let prompt_layers = PromptLayers {
                identity: "You are a pragmatic coding assistant in a Rust CLI harness.".to_string(),
                channel: format!("target={}", delivery_target_label(&target)),
                skills: self.skills.summary_for_prompt(),
                memory: self.prompt_memory_layer(),
            };
            let system_prompt = assemble_prompt(&prompt_layers);
            let request_history = self.history_with_system_prompt(system_prompt);
            self.cancel_requested.store(false, Ordering::Relaxed);
            if cli_reasoning {
                self.overlay_push("I'm calling the model for the next action.")?;
                self.overlay_start_working()?;
            }

            let mut streamed_assistant = String::new();
            let mut on_event = |event: ModelRuntimeEvent| match event {
                ModelRuntimeEvent::ReasoningSummaryDelta(text) => {
                    let preview = summarize_reasoning_preview(&text, 140);
                    let _ = self.overlay_push(format!("thinking: {preview}"));
                }
                ModelRuntimeEvent::OutputTextDelta(delta) => {
                    streamed_assistant.push_str(&delta);
                    let preview = summarize_reasoning_preview(&streamed_assistant, 140);
                    let _ = self.overlay_push(format!("drafting: {preview}"));
                }
            };

            let working_done = Arc::new(AtomicBool::new(false));
            let worker_join = if cli_reasoning {
                let done = Arc::clone(&working_done);
                let overlay = Arc::clone(&self.cli_reasoning_overlay);
                Some(thread::spawn(move || {
                    let started = Instant::now();
                    while !done.load(Ordering::Relaxed) {
                        if let Ok(mut overlay) = overlay.lock() {
                            let _ = overlay.tick_working(started.elapsed());
                        }
                        thread::sleep(Duration::from_millis(220));
                    }
                }))
            } else {
                None
            };

            let response_result = self.model.respond(
                &request_history,
                &specs,
                self.cancel_requested.as_ref(),
                &mut on_event,
            );
            working_done.store(true, Ordering::Relaxed);
            if let Some(join) = worker_join {
                let _ = join.join();
            }
            if cli_reasoning {
                self.overlay_stop_working()?;
            }

            let response = match response_result {
                Ok(output) => output,
                Err(err) => {
                    if cli_reasoning && err.to_string().contains("canceled by user") {
                        self.overlay_push("I stopped the current model run.")?;
                    }
                    let reply = format!("model error -> {err}");
                    self.emit_assistant_output(target.clone(), reply)?;
                    return Ok(());
                }
            };

            if cli_reasoning {
                self.emit_model_metrics(&response);
            }

            match &response.output {
                ModelOutput::Text(reply) => {
                    if cli_reasoning {
                        self.overlay_push("I'm finalizing the response.")?;
                    }
                    self.emit_reasoning_summaries(&response, &target)?;
                    self.emit_assistant_output(target.clone(), reply.clone())?;
                    return Ok(());
                }
                ModelOutput::ToolCalls(calls) => {
                    if calls.is_empty() {
                        let reply = "model returned an empty tool call set".to_string();
                        self.emit_assistant_output(target.clone(), reply)?;
                        return Ok(());
                    }
                    if cli_reasoning {
                        self.overlay_push(format!(
                            "I'm executing {} requested tool call(s).",
                            calls.len()
                        ))?;
                    }

                    for call in calls {
                        if cli_reasoning {
                            self.overlay_push(format!(
                                "I'm running tool `{}`.",
                                call.name.as_str()
                            ))?;
                        }
                        let output = self.execute_tool(call.name.as_str(), call.args.as_str())?;
                        self.push_message(Role::Tool, format!("{} {}", call.name, call.args))?;
                        let rendered = format!("tool:{} -> {}", call.name, output);
                        self.emit_assistant_output(target.clone(), rendered)?;
                    }
                }
            }
        }

        let reply = "stopped after max tool-call rounds".to_string();
        self.emit_assistant_output(target, reply)?;
        Ok(())
    }

    fn is_model_busy(&self) -> bool {
        self.model_busy.load(Ordering::Relaxed)
    }

    fn request_cancel(&self) -> bool {
        if !self.is_model_busy() {
            return false;
        }
        self.cancel_requested.store(true, Ordering::Relaxed);
        true
    }

    fn overlay_push(&self, message: impl Into<String>) -> Result<()> {
        if let Ok(mut overlay) = self.cli_reasoning_overlay.lock() {
            overlay.push(message)?;
        }
        Ok(())
    }

    fn overlay_clear(&self) -> Result<()> {
        if let Ok(mut overlay) = self.cli_reasoning_overlay.lock() {
            overlay.clear()?;
        }
        Ok(())
    }

    fn overlay_set_enabled(&self, enabled: bool) -> Result<()> {
        if let Ok(mut overlay) = self.cli_reasoning_overlay.lock() {
            overlay.set_enabled(enabled)?;
        }
        Ok(())
    }

    fn overlay_start_working(&self) -> Result<()> {
        if let Ok(mut overlay) = self.cli_reasoning_overlay.lock() {
            overlay.start_working()?;
        }
        Ok(())
    }

    fn overlay_stop_working(&self) -> Result<()> {
        if let Ok(mut overlay) = self.cli_reasoning_overlay.lock() {
            overlay.stop_working()?;
        }
        Ok(())
    }

    fn emit_model_metrics(&self, response: &ModelResponse) {
        if let Some(usage) = &response.usage {
            println!(
                "metrics> input_tokens={} output_tokens={} reasoning_tokens={} total_tokens={}",
                usage.input_tokens, usage.output_tokens, usage.reasoning_tokens, usage.total_tokens
            );
        }
    }

    fn emit_reasoning_summaries(
        &mut self,
        response: &ModelResponse,
        target: &DeliveryTarget,
    ) -> Result<()> {
        if *target != DeliveryTarget::Cli {
            return Ok(());
        }
        for summary in &response.reasoning_summaries {
            if summary.trim().is_empty() {
                continue;
            }
            let line = summarize_reasoning_preview(summary, 200);
            println!("reasoning> {line}");
        }
        Ok(())
    }

    fn history_with_system_prompt(&self, system_prompt: String) -> Vec<Message> {
        let mut history = Vec::with_capacity(self.history.len() + 1);
        history.push(Message::new(Role::System, system_prompt));
        history.extend(self.history.clone());
        history
    }

    fn prompt_memory_layer(&self) -> String {
        match self.memory_worker.latest(3) {
            Ok(rows) if !rows.is_empty() => rows
                .into_iter()
                .map(|row| format!("- {}", row.summary))
                .collect::<Vec<_>>()
                .join("\n"),
            Ok(_) => "No long-term memory yet.".to_string(),
            Err(err) => format!("memory unavailable: {err}"),
        }
    }

    pub fn set_source_route(&mut self, source: &EventSource, agent_id: impl Into<String>) {
        self.router.set_route(source, agent_id);
    }

    pub fn register_agent_handler<F>(&mut self, agent_id: impl Into<String>, handler: F)
    where
        F: Fn(&mut Harness<M>, DeliveryTarget, DispatchTask) -> Result<()> + Send + Sync + 'static,
    {
        self.agent_handlers
            .insert(agent_id.into(), Arc::new(handler) as AgentHandler<M>);
    }

    fn register_builtin_agent_handlers(&mut self) {
        self.register_agent_handler("default", |harness, target, _task| {
            harness.run_default_agent_turn(target)
        });
        self.register_agent_handler("planner", |harness, target, task| {
            let plan = harness.planner.create_plan(&task.content)?;
            let eval = harness.planner.evaluate_plan(&plan)?;
            harness.emit_assistant_output(
                target,
                format!(
                    "planner produced {} with {} step(s), eval={}.",
                    plan.id,
                    plan.steps.len(),
                    eval.status
                ),
            )
        });
    }

    fn run_default_agent_turn(&mut self, target: DeliveryTarget) -> Result<()> {
        self.run_model_turn(target)
    }

    fn dispatch_task(
        &mut self,
        source: &EventSource,
        target: DeliveryTarget,
        task: DispatchTask,
        websocket_session_id: Option<String>,
        telegram_chat_id: Option<i64>,
    ) -> Result<()> {
        if let Some(session_id) = websocket_session_id.as_deref() {
            let owner = self.event_bus.websocket_session_owner(session_id)?;
            if owner.is_none() {
                self.event_bus
                    .upsert_websocket_session_owner(session_id, task.to_agent.as_str())?;
            }
            let allowed = self
                .event_bus
                .websocket_session_agent_allowed(session_id, task.to_agent.as_str())?;
            if !allowed {
                self.emit_assistant_output(
                    DeliveryTarget::WebSocket,
                    format!(
                        "dispatch denied by websocket ACL: session={} agent={}",
                        session_id, task.to_agent
                    ),
                )?;
                return Ok(());
            }
        }

        if let Some(chat_id) = telegram_chat_id {
            let owner = self.event_bus.telegram_chat_owner(chat_id)?;
            if owner.is_none() {
                self.event_bus
                    .upsert_telegram_chat_owner(chat_id, task.to_agent.as_str())?;
            }
            let allowed = self
                .event_bus
                .telegram_chat_agent_allowed(chat_id, task.to_agent.as_str())?;
            if !allowed {
                self.emit_assistant_output(
                    DeliveryTarget::Telegram,
                    format!(
                        "dispatch denied by telegram ACL: chat_id={} agent={}",
                        chat_id, task.to_agent
                    ),
                )?;
                return Ok(());
            }
        }

        let action = format!("dispatch:{}", task.to_agent);
        let context = PolicyContext::new(
            Some(source_label(source)),
            Some(delivery_target_label(&target)),
            Some(task.to_agent.as_str()),
        );
        let decision = self.policy_engine.evaluate(&action, &context);
        self.policy_engine.audit(&action, &context, &decision)?;
        if !decision.allowed {
            self.emit_assistant_output(
                target,
                format!(
                    "dispatch denied by policy: {} ({})",
                    task.to_agent, decision.reason
                ),
            )?;
            return Ok(());
        }

        let handler = self
            .agent_handlers
            .get(task.to_agent.as_str())
            .cloned()
            .ok_or_else(|| anyhow!("no handler registered for agent `{}`", task.to_agent))?;
        let previous_context = self.active_policy_context.clone();
        self.active_policy_context = ActivePolicyContext {
            source: source.clone(),
            channel: target.clone(),
            agent: task.to_agent.clone(),
            websocket_session_id,
            telegram_chat_id,
        };
        let result = handler(self, target, task);
        self.active_policy_context = previous_context;
        result
    }

    fn process_inbound_events(&mut self) -> Result<()> {
        let mut processed = 0usize;
        while processed < self.max_inbound_events_per_cycle {
            let Some(event) = self.event_bus.pop_inbound()? else {
                break;
            };
            let target = delivery_target_for_source(&event.source);
            if let Err(err) = self.handle_inbound_event(target.clone(), event) {
                self.emit_assistant_output(target, format!("event processing error -> {err}"))?;
            }
            processed += 1;
        }
        Ok(())
    }

    fn handle_inbound_event(&mut self, target: DeliveryTarget, event: InboundEvent) -> Result<()> {
        match event.payload {
            InboundPayload::UserMessage(content) => {
                self.push_message(Role::User, content.clone())?;
                let route = self.router.route(&event.source).to_string();
                self.dispatch_task(
                    &event.source,
                    target,
                    DispatchTask::new("ingress", route, content),
                    None,
                    None,
                )
            }
            InboundPayload::WebSocketMessage {
                session_id,
                content,
            } => {
                self.push_message(Role::User, content.clone())?;
                let route = match self
                    .event_bus
                    .websocket_session_owner(session_id.as_str())?
                {
                    Some(owner) => owner,
                    None => {
                        let owner = self.router.route(&event.source).to_string();
                        self.event_bus
                            .upsert_websocket_session_owner(session_id.as_str(), owner.as_str())?;
                        owner
                    }
                };
                self.dispatch_task(
                    &event.source,
                    target,
                    DispatchTask::new("ingress", route, content),
                    Some(session_id),
                    None,
                )
            }
            InboundPayload::TelegramMessage { chat_id, content } => {
                self.push_message(Role::User, content.clone())?;
                let route = match self.event_bus.telegram_chat_owner(chat_id)? {
                    Some(owner) => owner,
                    None => {
                        let owner = self.router.route(&event.source).to_string();
                        self.event_bus
                            .upsert_telegram_chat_owner(chat_id, owner.as_str())?;
                        owner
                    }
                };
                self.dispatch_task(
                    &event.source,
                    target,
                    DispatchTask::new("ingress", route, content),
                    None,
                    Some(chat_id),
                )
            }
            InboundPayload::DispatchTask(task) => {
                self.dispatch_task(&event.source, target, task, None, None)
            }
        }
    }

    fn handle_slash(&mut self, input: &str) -> Result<bool> {
        let mut parts = input.splitn(2, char::is_whitespace);
        let command = parts.next().unwrap_or_default();
        let args = parts.next().unwrap_or("").trim();

        match command {
            "/help" => {
                self.print_help();
                Ok(true)
            }
            "/tools" => {
                for (name, desc, privileged) in self.tools.list() {
                    let level = if privileged { "privileged" } else { "safe" };
                    println!("{name} [{level}] - {desc}");
                }
                Ok(true)
            }
            "/compact" => {
                self.compact_history();
                println!("Compacted in-memory history.");
                Ok(true)
            }
            "/approve" => match args {
                "off" => {
                    self.permissions.ask_before_privileged_tools = false;
                    println!("Privileged tool approvals disabled.");
                    Ok(true)
                }
                "on" => {
                    self.permissions.ask_before_privileged_tools = true;
                    println!("Privileged tool approvals enabled.");
                    Ok(true)
                }
                _ => bail!("usage: /approve <on|off>"),
            },
            "/thinking" => match args {
                "on" => {
                    self.overlay_set_enabled(true)?;
                    println!("CLI reasoning overlay enabled.");
                    Ok(true)
                }
                "off" => {
                    self.overlay_set_enabled(false)?;
                    println!("CLI reasoning overlay disabled.");
                    Ok(true)
                }
                _ => bail!("usage: /thinking <on|off>"),
            },
            "/cancel" => {
                if self.request_cancel() {
                    println!("Cancellation requested for current model run.");
                } else {
                    println!("No active model run to cancel.");
                }
                Ok(true)
            }
            "/route" => {
                self.handle_route_command(args)?;
                Ok(true)
            }
            "/dispatch" => {
                self.handle_dispatch_command(args)?;
                Ok(true)
            }
            "/ws" => {
                self.handle_websocket_command(args)?;
                Ok(true)
            }
            "/tg" => {
                self.handle_telegram_command(args)?;
                Ok(true)
            }
            "/reload-config" => {
                self.reload_config_forced()?;
                println!("Config reloaded.");
                Ok(true)
            }
            "/skills" => {
                self.handle_skills_command(args)?;
                Ok(true)
            }
            "/policy" => {
                self.handle_policy_command(args)?;
                Ok(true)
            }
            "/plan" => {
                self.handle_plan_command(args)?;
                Ok(true)
            }
            "/tick" => {
                self.tick_scheduler_once()?;
                self.process_inbound_events()?;
                self.flush_outbound_events()?;
                Ok(true)
            }
            "/post" => {
                self.handle_post_command(args)?;
                Ok(true)
            }
            "/memory" => {
                self.handle_memory_command(args)?;
                Ok(true)
            }
            "/exit" | "/quit" => {
                std::process::exit(0);
            }
            _ => Ok(false),
        }
    }

    fn handle_route_command(&mut self, args: &str) -> Result<()> {
        let mut parts = args.splitn(2, char::is_whitespace);
        let source = parts.next().unwrap_or("").trim();
        let agent_id = parts.next().unwrap_or("").trim();

        if source.is_empty() || agent_id.is_empty() {
            bail!("usage: /route <source> <agent_id>");
        }

        let source = parse_event_source(source)?;
        self.set_source_route(&source, agent_id.to_string());
        println!("Route updated: {} -> {}", source_label(&source), agent_id);
        Ok(())
    }

    fn handle_dispatch_command(&mut self, args: &str) -> Result<()> {
        let mut parts = args.splitn(2, char::is_whitespace);
        let to_agent = parts.next().unwrap_or("").trim();
        let content = parts.next().unwrap_or("").trim();

        if to_agent.is_empty() || content.is_empty() {
            bail!("usage: /dispatch <to_agent> <content>");
        }

        self.event_bus.publish_inbound(InboundEvent {
            source: EventSource::Cli,
            payload: InboundPayload::DispatchTask(DispatchTask::new("cli", to_agent, content)),
        })?;
        self.process_inbound_events()?;
        self.flush_outbound_events()?;
        self.auto_compact_if_needed();
        Ok(())
    }

    fn handle_websocket_command(&mut self, args: &str) -> Result<()> {
        let mut parts = args.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").trim();
        let rest = parts.next().unwrap_or("").trim();

        match subcommand {
            "send" => {
                if rest.is_empty() {
                    bail!("usage: /ws send <content>");
                }
                self.event_bus
                    .publish_inbound(InboundEvent::websocket_message("ws-cli", rest))?;
                self.process_inbound_events()?;
                self.flush_outbound_events()?;
                self.auto_compact_if_needed();
                Ok(())
            }
            "broadcast" => {
                if rest.is_empty() {
                    bail!("usage: /ws broadcast <content>");
                }
                self.websocket_server.send(None, rest);
                println!("[ws:broadcast] sent");
                Ok(())
            }
            "poll" => {
                match self.websocket_outbox.pop_front() {
                    Some(msg) => println!("[ws] {msg}"),
                    None => println!("[ws] <empty>"),
                }
                Ok(())
            }
            "clients" => {
                println!(
                    "connected websocket clients: {}",
                    self.websocket_server.client_count()
                );
                Ok(())
            }
            "bind" => {
                let mut inner = rest.splitn(2, char::is_whitespace);
                let session_id = inner.next().unwrap_or("").trim();
                let agent_id = inner.next().unwrap_or("").trim();
                if session_id.is_empty() || agent_id.is_empty() {
                    bail!("usage: /ws bind <session_id> <agent_id>");
                }
                self.event_bus
                    .upsert_websocket_session_owner(session_id, agent_id)?;
                println!("ws session owner set: {} -> {}", session_id, agent_id);
                Ok(())
            }
            "allow" => {
                let mut inner = rest.splitn(2, char::is_whitespace);
                let session_id = inner.next().unwrap_or("").trim();
                let agent_id = inner.next().unwrap_or("").trim();
                if session_id.is_empty() || agent_id.is_empty() {
                    bail!("usage: /ws allow <session_id> <agent_id>");
                }
                self.event_bus
                    .allow_websocket_session_agent(session_id, agent_id)?;
                println!("ws session ACL allow: {} -> {}", session_id, agent_id);
                Ok(())
            }
            "acl" => {
                let session_id = rest;
                if session_id.is_empty() {
                    bail!("usage: /ws acl <session_id>");
                }
                let owner = self
                    .event_bus
                    .websocket_session_owner(session_id)?
                    .unwrap_or_else(|| "<none>".to_string());
                let acl = self.event_bus.websocket_session_acl(session_id)?;
                if acl.is_empty() {
                    println!("ws session {} owner={} acl=<empty>", session_id, owner);
                } else {
                    println!(
                        "ws session {} owner={} acl={}",
                        session_id,
                        owner,
                        acl.join(",")
                    );
                }
                Ok(())
            }
            "rotate-token" => {
                if rest.is_empty() {
                    bail!("usage: /ws rotate-token <new_token>");
                }
                self.websocket_server.rotate_auth_token(rest);
                println!("websocket auth token rotated (grace period in effect)");
                Ok(())
            }
            "status" => {
                let snapshot = self.websocket_server.policy_snapshot();
                println!(
                    "ws policy auth_required={} current_token={} previous_token={} max_clients={} idle_timeout_secs={} ping_interval_secs={} auth_rotation_grace_secs={}",
                    snapshot.auth_required,
                    snapshot.has_current_token,
                    snapshot.has_previous_token,
                    snapshot.max_clients,
                    snapshot.idle_timeout_secs,
                    snapshot.ping_interval_secs,
                    snapshot.auth_rotation_grace_secs
                );
                Ok(())
            }
            _ => bail!(
                "usage: /ws <send|broadcast|poll|clients|bind|allow|acl|status|rotate-token> ..."
            ),
        }
    }

    fn handle_telegram_command(&mut self, args: &str) -> Result<()> {
        let mut parts = args.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").trim();
        let rest = parts.next().unwrap_or("").trim();

        match subcommand {
            "send" => {
                let mut inner = rest.splitn(2, char::is_whitespace);
                let chat_id_raw = inner.next().unwrap_or("").trim();
                let content = inner.next().unwrap_or("").trim();
                if chat_id_raw.is_empty() || content.is_empty() {
                    bail!("usage: /tg send <chat_id> <content>");
                }
                let chat_id: i64 = chat_id_raw
                    .parse()
                    .map_err(|_| anyhow!("chat_id must be a valid integer"))?;
                self.event_bus
                    .publish_inbound(InboundEvent::telegram_message(chat_id, content))?;
                self.process_inbound_events()?;
                self.flush_outbound_events()?;
                self.auto_compact_if_needed();
                Ok(())
            }
            "poll" => {
                match self.telegram_outbox.pop_front() {
                    Some(msg) => println!("[tg] {msg}"),
                    None => println!("[tg] <empty>"),
                }
                Ok(())
            }
            "bind" => {
                let mut inner = rest.splitn(2, char::is_whitespace);
                let chat_id_raw = inner.next().unwrap_or("").trim();
                let agent_id = inner.next().unwrap_or("").trim();
                if chat_id_raw.is_empty() || agent_id.is_empty() {
                    bail!("usage: /tg bind <chat_id> <agent_id>");
                }
                let chat_id: i64 = chat_id_raw
                    .parse()
                    .map_err(|_| anyhow!("chat_id must be a valid integer"))?;
                self.event_bus
                    .upsert_telegram_chat_owner(chat_id, agent_id)?;
                println!("tg chat owner set: {} -> {}", chat_id, agent_id);
                Ok(())
            }
            "allow" => {
                let mut inner = rest.splitn(2, char::is_whitespace);
                let chat_id_raw = inner.next().unwrap_or("").trim();
                let agent_id = inner.next().unwrap_or("").trim();
                if chat_id_raw.is_empty() || agent_id.is_empty() {
                    bail!("usage: /tg allow <chat_id> <agent_id>");
                }
                let chat_id: i64 = chat_id_raw
                    .parse()
                    .map_err(|_| anyhow!("chat_id must be a valid integer"))?;
                self.event_bus
                    .allow_telegram_chat_agent(chat_id, agent_id)?;
                println!("tg chat ACL allow: {} -> {}", chat_id, agent_id);
                Ok(())
            }
            "acl" => {
                if rest.is_empty() {
                    bail!("usage: /tg acl <chat_id>");
                }
                let chat_id: i64 = rest
                    .parse()
                    .map_err(|_| anyhow!("chat_id must be a valid integer"))?;
                let owner = self
                    .event_bus
                    .telegram_chat_owner(chat_id)?
                    .unwrap_or_else(|| "<none>".to_string());
                let acl = self.event_bus.telegram_chat_acl(chat_id)?;
                if acl.is_empty() {
                    println!("tg chat {} owner={} acl=<empty>", chat_id, owner);
                } else {
                    println!("tg chat {} owner={} acl={}", chat_id, owner, acl.join(","));
                }
                Ok(())
            }
            "status" => {
                let snapshot = self.telegram_runtime.policy_snapshot();
                println!(
                    "tg policy enabled={} has_bot_token={} api_base_url={} poll_interval_secs={} poll_timeout_secs={} allowed_chat_ids={} outbox_pending={}",
                    snapshot.enabled,
                    snapshot.has_bot_token,
                    snapshot.api_base_url,
                    snapshot.poll_interval_secs,
                    snapshot.poll_timeout_secs,
                    snapshot.allowed_chat_ids_count,
                    self.telegram_outbox.len()
                );
                Ok(())
            }
            _ => bail!("usage: /tg <send|poll|bind|allow|acl|status> ..."),
        }
    }

    fn handle_skills_command(&mut self, args: &str) -> Result<()> {
        match args {
            "reload" => {
                let cfg = self.config_manager.load()?;
                let path = default_skill_dir(&self.tool_ctx.workspace_root, &cfg.skill_dir);
                self.skills = SkillRegistry::load_from_dir(path)?;
                println!("skills reloaded ({})", self.skills.skills.len());
                Ok(())
            }
            "" | "list" => {
                let names = self.skills.names();
                if names.is_empty() {
                    println!("No skills loaded.");
                } else {
                    for name in names {
                        println!("{name}");
                    }
                }
                Ok(())
            }
            _ => bail!("usage: /skills [list|reload]"),
        }
    }

    fn handle_policy_command(&mut self, args: &str) -> Result<()> {
        match args {
            "reload" => {
                self.policy_engine.reload()?;
                println!("policy reloaded");
                Ok(())
            }
            _ => bail!("usage: /policy reload"),
        }
    }

    fn handle_plan_command(&mut self, args: &str) -> Result<()> {
        let trimmed = args.trim();
        if trimmed.is_empty() {
            bail!(
                "usage: /plan <goal> | /plan status <plan_id> | /plan resume <plan_id> | /plan start <plan_id> <step_id> | /plan done <plan_id> <step_id> | /plan fail <plan_id> <step_id> <reason>"
            );
        }

        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or_default();
        let rest = parts.next().unwrap_or("").trim();

        match subcommand {
            "status" => {
                if rest.is_empty() {
                    bail!("usage: /plan status <plan_id>");
                }
                let plan = self.planner.load_plan(rest)?;
                let eval = self.planner.evaluate_plan(&plan)?;
                println!(
                    "plan={} steps={} eval={} findings={}",
                    plan.id,
                    plan.steps.len(),
                    eval.status,
                    eval.findings.join("; ")
                );
                Ok(())
            }
            "resume" => {
                if rest.is_empty() {
                    bail!("usage: /plan resume <plan_id>");
                }
                match self.planner.resume_plan(rest)? {
                    Some(step) => println!(
                        "next step: id={} status={:?} deps={:?} desc={}",
                        step.id, step.status, step.depends_on, step.description
                    ),
                    None => println!("no resumable step"),
                }
                Ok(())
            }
            "start" | "done" => {
                let mut inner = rest.splitn(3, char::is_whitespace);
                let plan_id = inner.next().unwrap_or("").trim();
                let step_id = inner.next().unwrap_or("").trim();
                if plan_id.is_empty() || step_id.is_empty() {
                    bail!("usage: /plan {subcommand} <plan_id> <step_id>");
                }
                let step_id: usize = step_id
                    .parse()
                    .map_err(|_| anyhow!("step_id must be a positive integer"))?;
                let plan = if subcommand == "start" {
                    self.planner.mark_step_in_progress(plan_id, step_id)?
                } else {
                    self.planner.mark_step_completed(plan_id, step_id)?
                };
                println!(
                    "updated plan={} step={} action={}",
                    plan.id, step_id, subcommand
                );
                Ok(())
            }
            "fail" => {
                let mut inner = rest.splitn(3, char::is_whitespace);
                let plan_id = inner.next().unwrap_or("").trim();
                let step_id = inner.next().unwrap_or("").trim();
                let reason = inner.next().unwrap_or("").trim();
                if plan_id.is_empty() || step_id.is_empty() || reason.is_empty() {
                    bail!("usage: /plan fail <plan_id> <step_id> <reason>");
                }
                let step_id: usize = step_id
                    .parse()
                    .map_err(|_| anyhow!("step_id must be a positive integer"))?;
                let plan = self.planner.mark_step_failed(plan_id, step_id, reason)?;
                println!("updated plan={} step={} action=fail", plan.id, step_id);
                Ok(())
            }
            _ => {
                let goal = trimmed;
                let plan = self.planner.create_plan(goal)?;
                let eval = self.planner.evaluate_plan(&plan)?;
                println!(
                    "plan={} steps={} eval={}",
                    plan.id,
                    plan.steps.len(),
                    eval.status
                );
                Ok(())
            }
        }
    }

    fn handle_post_command(&mut self, args: &str) -> Result<()> {
        let mut parts = args.splitn(2, char::is_whitespace);
        let target = parts.next().unwrap_or("").trim();
        let content = parts.next().unwrap_or("").trim();

        if target.is_empty() || content.is_empty() {
            bail!("usage: /post <cli|file|websocket|telegram> <content>");
        }

        let target = parse_delivery_target(target)?;
        let action = format!("outbound:{}", delivery_target_label(&target));
        let context = PolicyContext::new(
            Some("cli"),
            Some(delivery_target_label(&target)),
            Some("default"),
        );
        let decision = self.policy_engine.evaluate(&action, &context);
        self.policy_engine.audit(&action, &context, &decision)?;
        if !decision.allowed {
            bail!("outbound denied by policy: {}", decision.reason);
        }
        self.emit_assistant_output(target, content.to_string())?;
        self.flush_outbound_events()?;
        Ok(())
    }

    fn handle_memory_command(&mut self, args: &str) -> Result<()> {
        let mut parts = args.splitn(2, char::is_whitespace);
        let subcommand = parts.next().unwrap_or("").trim();
        let rest = parts.next().unwrap_or("").trim();

        match subcommand {
            "latest" => {
                for row in self.memory_worker.latest(5)? {
                    println!("{}", row.summary);
                }
                Ok(())
            }
            "search" => {
                if rest.is_empty() {
                    bail!("usage: /memory search <query>");
                }
                for row in self.memory_worker.search(rest, 5)? {
                    println!("[score={:.2}] {}", row.score, row.summary);
                }
                Ok(())
            }
            "compact" => {
                let stats = self.memory_worker.compact()?;
                println!(
                    "memory compacted: ttl_removed={} overflow_removed={} remaining={}",
                    stats.removed_ttl, stats.removed_overflow, stats.remaining
                );
                Ok(())
            }
            "maintain" => {
                let stats = self
                    .memory_worker
                    .maintain_ann_index(self.memory_ann_backfill_batch_size)?;
                println!(
                    "memory ann maintenance: scanned={} backfilled={} reindexed={} remaining_unindexed={}",
                    stats.scanned_rows,
                    stats.backfilled_rows,
                    stats.reindexed_rows,
                    stats.remaining_unindexed
                );
                Ok(())
            }
            "status" => {
                let diag = self.memory_worker.diagnostics()?;
                println!(
                    "memory retrieval: searches={} ann_attempts={} ann_hits={} ann_fallbacks={} scan_only={} ann_hit_rate={:.1}% ann_fallback_rate={:.1}% scan_only_rate={:.1}%",
                    diag.search_total,
                    diag.ann_attempt_total,
                    diag.ann_hit_total,
                    diag.ann_fallback_total,
                    diag.scan_only_total,
                    diag.ann_hit_rate * 100.0,
                    diag.ann_fallback_rate * 100.0,
                    diag.scan_only_rate * 100.0
                );
                println!(
                    "memory retrieval quality: candidates_total={} candidates_avg={:.1} candidates_p95={} latency_avg_ms={:.2} latency_p95_ms={}",
                    diag.candidate_examined_total,
                    diag.candidate_examined_avg,
                    diag.candidate_examined_p95,
                    diag.query_latency_avg_ms,
                    diag.query_latency_p95_ms
                );
                println!(
                    "memory ann lag: remaining_unindexed={} cursor={}/{} progress={:.1}% last_maintained_at={}",
                    diag.remaining_unindexed,
                    diag.ann_cursor,
                    diag.ann_cursor_max,
                    diag.ann_cursor_progress * 100.0,
                    diag.ann_last_maintained_at.as_deref().unwrap_or("never")
                );
                Ok(())
            }
            "snapshot" => {
                self.print_memory_status_snapshot()?;
                Ok(())
            }
            _ => bail!("usage: /memory <latest|search ...|compact|maintain|status|snapshot>"),
        }
    }

    fn tick_scheduler_once(&mut self) -> Result<()> {
        enqueue_heartbeat(&mut self.event_bus)?;
        Ok(())
    }

    fn reload_config_forced(&mut self) -> Result<()> {
        let cfg = self.config_manager.load()?;
        self.apply_config(cfg)
    }

    fn reload_config_if_changed(&mut self) -> Result<()> {
        if let Some(cfg) = self.config_manager.reload_if_changed()? {
            self.apply_config(cfg)?;
        }
        Ok(())
    }

    fn apply_config(&mut self, cfg: HarnessConfig) -> Result<()> {
        self.compact_threshold_messages = cfg.compact_threshold_messages;
        self.max_inbound_events_per_cycle = cfg.max_inbound_events_per_cycle;
        self.max_outbound_events_per_cycle = cfg.max_outbound_events_per_cycle;
        self.scheduler_control
            .set_heartbeat_interval_secs(cfg.heartbeat_interval_secs);
        self.scheduler_control.set_enabled(cfg.scheduler_enabled);
        self.scheduler_control
            .set_max_pending_inbound(cfg.max_scheduler_pending_inbound);
        self.memory_worker.set_retention(
            cfg.memory_retention_ttl_days,
            cfg.memory_retention_max_records,
        );
        self.memory_worker.set_search_profile(
            cfg.memory_search_mode.as_str(),
            cfg.memory_hybrid_lexical_weight,
            cfg.memory_hybrid_semantic_weight,
        );
        self.memory_worker.set_embedding_backend(
            cfg.memory_embedding_provider.as_str(),
            cfg.memory_embedding_model.as_str(),
            cfg.memory_embedding_base_url.as_str(),
            cfg.memory_embedding_api_key_env.as_str(),
            cfg.memory_embedding_dimensions,
        );
        self.memory_worker.set_ann_profile(
            cfg.memory_ann_enabled,
            cfg.memory_ann_scan_threshold,
            cfg.memory_ann_probe_count,
            cfg.memory_ann_candidate_cap,
        );
        self.memory_ann_backfill_batch_size = cfg.memory_ann_backfill_batch_size.max(1);
        self.memory_ann_maintenance_interval_ticks = cfg.memory_ann_maintenance_interval_ticks;
        self.memory_status_snapshot_interval_ticks = cfg.memory_status_snapshot_interval_ticks;
        self.websocket_server
            .reconfigure(websocket_runtime_config_from_harness(&cfg));
        self.telegram_runtime
            .reconfigure(telegram_runtime_config_from_harness(&cfg));

        let skill_dir = default_skill_dir(&self.tool_ctx.workspace_root, &cfg.skill_dir);
        self.skills = SkillRegistry::load_from_dir(skill_dir)?;
        self.policy_engine.reload()?;
        Ok(())
    }

    fn execute_tool(&mut self, name: &str, args: &str) -> Result<String> {
        let Some(privileged) = self.tools.get(name).map(|t| t.privileged()) else {
            return Ok(format!("Unknown tool: {name}. See `/tools`."));
        };

        let action = format!("tool:{name}");
        let context = self.current_policy_context();
        let decision = self.policy_engine.evaluate(&action, &context);
        self.policy_engine.audit(&action, &context, &decision)?;
        if !decision.allowed {
            return Ok(format!(
                "Tool call denied by policy: {name} ({})",
                decision.reason
            ));
        }

        let needs_approval = self.permissions.should_require_approval(name, privileged);
        if needs_approval && !self.prompt_approval(name)? {
            return Ok(format!("Tool call denied: {name}"));
        }

        let Some(tool) = self.tools.get(name) else {
            return Ok(format!("Unknown tool: {name}. See `/tools`."));
        };

        match tool.run(args, &self.tool_ctx) {
            Ok(output) => Ok(output),
            Err(err) => Ok(format!("tool:{name} error -> {err}")),
        }
    }

    fn prompt_approval(&mut self, tool_name: &str) -> Result<bool> {
        print!("Approve privileged tool `{tool_name}`? [y/N/a=always] ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let normalized = input.trim().to_lowercase();
        match normalized.as_str() {
            "y" | "yes" => Ok(true),
            "a" | "always" => {
                self.permissions.approve_tool(tool_name.to_string());
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn push_message(&mut self, role: Role, content: String) -> Result<()> {
        let msg = Message::new(role, content);
        self.session.append(&msg)?;
        self.history.push(msg);
        let _ = self.memory_worker.maybe_snapshot(&self.history)?;
        Ok(())
    }

    fn run_memory_maintenance_tick(&mut self) -> Result<()> {
        self.runtime_tick_count = self.runtime_tick_count.saturating_add(1);
        let interval = self.memory_ann_maintenance_interval_ticks.max(1);
        if self.runtime_tick_count % interval == 0 {
            let _ = self
                .memory_worker
                .maintain_ann_index(self.memory_ann_backfill_batch_size.max(1))?;
        }
        let status_interval = self.memory_status_snapshot_interval_ticks;
        if status_interval > 0 && self.runtime_tick_count % status_interval == 0 {
            self.print_memory_status_snapshot()?;
        }
        Ok(())
    }

    fn print_memory_status_snapshot(&self) -> Result<()> {
        let diag = self.memory_worker.diagnostics()?;
        println!(
            "memory snapshot: searches={} ann_hit={} fallback={} scan_only={} cand_p95={} latency_p95_ms={} remaining_unindexed={} cursor_progress={:.1}%",
            diag.search_total,
            diag.ann_hit_total,
            diag.ann_fallback_total,
            diag.scan_only_total,
            diag.candidate_examined_p95,
            diag.query_latency_p95_ms,
            diag.remaining_unindexed,
            diag.ann_cursor_progress * 100.0
        );
        Ok(())
    }

    fn emit_assistant_output(&mut self, target: DeliveryTarget, content: String) -> Result<()> {
        self.push_message(Role::Assistant, content.clone())?;
        let event = match target {
            DeliveryTarget::WebSocket => {
                if let Some(session) = self.active_policy_context.websocket_session_id.clone() {
                    OutboundEvent::with_websocket_session(target, content, session)
                } else {
                    OutboundEvent::new(target, content)
                }
            }
            DeliveryTarget::Telegram => {
                if let Some(chat_id) = self.active_policy_context.telegram_chat_id {
                    OutboundEvent::with_telegram_chat(target, content, chat_id)
                } else {
                    OutboundEvent::new(target, content)
                }
            }
            _ => OutboundEvent::new(target, content),
        };
        self.event_bus.publish_outbound(event)?;
        Ok(())
    }

    fn current_policy_context(&self) -> PolicyContext {
        PolicyContext::new(
            Some(source_label(&self.active_policy_context.source)),
            Some(delivery_target_label(&self.active_policy_context.channel)),
            Some(self.active_policy_context.agent.as_str()),
        )
    }

    fn flush_outbound_events(&mut self) -> Result<()> {
        let mut delivered = 0usize;
        while delivered < self.max_outbound_events_per_cycle {
            let Some(pending) = self.event_bus.next_outbound_for_delivery()? else {
                break;
            };
            match self.deliver_outbound(&pending.event) {
                Ok(()) => {
                    self.event_bus.mark_outbound_processed(pending.id)?;
                    delivered += 1;
                }
                Err(err) => {
                    self.event_bus
                        .mark_outbound_delivery_failed(pending.id, &format!("{err:#}"))?;
                    break;
                }
            }
        }
        Ok(())
    }

    fn deliver_outbound(&mut self, event: &OutboundEvent) -> Result<()> {
        match event.target {
            DeliveryTarget::Cli => self.deliver_cli(event.content.as_str()),
            DeliveryTarget::File => self.deliver_file(event.content.as_str()),
            DeliveryTarget::WebSocket => self.deliver_websocket(
                event.content.as_str(),
                event.websocket_session_id.as_deref(),
            ),
            DeliveryTarget::Telegram => {
                self.deliver_telegram(event.content.as_str(), event.telegram_chat_id)
            }
        }
    }

    fn deliver_cli(&mut self, content: &str) -> Result<()> {
        self.overlay_clear()?;
        println!("{}", render_cli_chat_line("assistant", content));
        Ok(())
    }

    fn deliver_file(&mut self, content: &str) -> Result<()> {
        append_line_jsonl(&self.file_outbox_path, content)
    }

    fn deliver_websocket(
        &mut self,
        content: &str,
        websocket_session_id: Option<&str>,
    ) -> Result<()> {
        self.websocket_outbox.push_back(content.to_string());
        if let Some(session_id) = websocket_session_id {
            self.websocket_server.send(Some(session_id), content);
        }
        Ok(())
    }

    fn deliver_telegram(&mut self, content: &str, telegram_chat_id: Option<i64>) -> Result<()> {
        self.telegram_outbox.push_back(content.to_string());
        let Some(chat_id) = telegram_chat_id else {
            bail!("telegram outbound missing chat_id");
        };
        self.telegram_runtime.send_message(chat_id, content)
    }

    fn ingest_websocket_inbound_events(&mut self) -> Result<()> {
        for inbound in self.websocket_server.drain_inbound() {
            self.event_bus
                .publish_inbound(InboundEvent::websocket_message(
                    inbound.session_id,
                    inbound.content,
                ))?;
        }
        Ok(())
    }

    fn ingest_telegram_inbound_events(&mut self) -> Result<()> {
        let inbound = match self.telegram_runtime.drain_inbound() {
            Ok(events) => events,
            Err(err) => {
                eprintln!("telegram poll failed: {err:#}");
                return Ok(());
            }
        };
        for event in inbound {
            self.event_bus
                .publish_inbound(InboundEvent::telegram_message(event.chat_id, event.content))?;
        }
        Ok(())
    }

    fn auto_compact_if_needed(&mut self) {
        if self.history.len() > self.compact_threshold_messages {
            self.compact_history();
        }
    }

    fn compact_history(&mut self) {
        if self.history.len() <= 12 {
            return;
        }
        let keep = self
            .history
            .split_off(self.history.len().saturating_sub(12));
        self.history = keep;
    }

    fn print_help(&self) {
        println!("Slash commands:");
        println!("  /help          Show this help.");
        println!("  /tools         List tools.");
        println!("  /compact       Compact in-memory history.");
        println!("  /route         Set source route (`/route <source> <agent_id>`).");
        println!("  /dispatch      Dispatch task (`/dispatch <to_agent> <content>`).");
        println!(
            "  /ws            WebSocket controls (`/ws send <content>`, `/ws broadcast <content>`, `/ws poll`, `/ws clients`, `/ws bind <sid> <agent>`, `/ws allow <sid> <agent>`, `/ws acl <sid>`, `/ws status`, `/ws rotate-token <token>`)."
        );
        println!(
            "  /tg            Telegram controls (`/tg send <chat_id> <content>`, `/tg poll`, `/tg bind <chat_id> <agent>`, `/tg allow <chat_id> <agent>`, `/tg acl <chat_id>`, `/tg status`)."
        );
        println!("  /reload-config Force config reload from .agent/config.json.");
        println!("  /skills        Skill operations (`/skills list`, `/skills reload`).");
        println!("  /policy        Policy operations (`/policy reload`).");
        println!(
            "  /plan          Plan workflow (`/plan <goal>`, `status|resume|start|done|fail`)."
        );
        println!("  /tick          Run scheduler tick now.");
        println!("  /post          Emit outbound event (`/post <channel> <content>`).");
        println!(
            "  /memory        Memory operations (`/memory latest`, `/memory search <q>`, `/memory compact`, `/memory maintain`, `/memory status`, `/memory snapshot`)."
        );
        println!("  /approve on    Require approvals for privileged tools.");
        println!("  /approve off   Disable approvals for privileged tools.");
        println!("  /thinking on   Enable transient CLI thinking overlay.");
        println!("  /thinking off  Disable transient CLI thinking overlay.");
        println!("  /cancel        Cancel the currently running model turn.");
        println!("  /exit          Exit.");
        println!("Tool call format: tool:<name> <args>");
    }
}

fn parse_tool_call(input: &str) -> Option<(&str, &str)> {
    let rest = input.strip_prefix("tool:")?;
    let mut parts = rest.splitn(2, char::is_whitespace);
    let name = parts.next()?.trim();
    if name.is_empty() {
        return None;
    }
    let args = parts.next().unwrap_or("").trim();
    Some((name, args))
}

fn delivery_target_for_source(source: &EventSource) -> DeliveryTarget {
    match source {
        EventSource::Cli => DeliveryTarget::Cli,
        EventSource::WebSocket => DeliveryTarget::WebSocket,
        EventSource::Telegram => DeliveryTarget::Telegram,
        EventSource::Scheduler => DeliveryTarget::Cli,
    }
}

fn parse_event_source(raw: &str) -> Result<EventSource> {
    match raw {
        "cli" => Ok(EventSource::Cli),
        "websocket" | "ws" => Ok(EventSource::WebSocket),
        "telegram" | "tg" => Ok(EventSource::Telegram),
        "scheduler" => Ok(EventSource::Scheduler),
        _ => bail!("unsupported source `{raw}` (supported: cli, websocket, telegram, scheduler)"),
    }
}

fn websocket_runtime_config_from_harness(cfg: &HarnessConfig) -> WebSocketRuntimeConfig {
    let ws_auth = if cfg.websocket_auth_required {
        let current = cfg.websocket_auth_token.trim();
        if current.is_empty() {
            WebSocketAuthConfig::disabled()
        } else {
            let previous = cfg.websocket_auth_previous_token.trim();
            if previous.is_empty() {
                WebSocketAuthConfig::token_required(current.to_string())
            } else {
                WebSocketAuthConfig::token_required_with_previous(
                    current.to_string(),
                    Some(previous.to_string()),
                )
            }
        }
    } else {
        WebSocketAuthConfig::disabled()
    };

    WebSocketRuntimeConfig {
        auth: ws_auth,
        max_clients: cfg.websocket_max_clients,
        idle_timeout_secs: cfg.websocket_idle_timeout_secs,
        ping_interval_secs: cfg.websocket_ping_interval_secs,
        auth_rotation_grace_secs: cfg.websocket_auth_rotation_grace_secs,
    }
}

fn telegram_runtime_config_from_harness(cfg: &HarnessConfig) -> TelegramRuntimeConfig {
    TelegramRuntimeConfig {
        enabled: cfg.telegram_enabled,
        bot_token: cfg.telegram_bot_token.clone(),
        api_base_url: cfg.telegram_api_base_url.clone(),
        poll_interval_secs: cfg.telegram_poll_interval_secs,
        poll_timeout_secs: cfg.telegram_poll_timeout_secs,
        allowed_chat_ids: cfg.telegram_allowed_chat_ids.clone(),
    }
}

fn parse_delivery_target(raw: &str) -> Result<DeliveryTarget> {
    match raw {
        "cli" => Ok(DeliveryTarget::Cli),
        "file" => Ok(DeliveryTarget::File),
        "websocket" | "ws" => Ok(DeliveryTarget::WebSocket),
        "telegram" | "tg" => Ok(DeliveryTarget::Telegram),
        _ => bail!("unsupported target `{raw}` (supported: cli, file, websocket, telegram)"),
    }
}

fn source_label(source: &EventSource) -> &'static str {
    match source {
        EventSource::Cli => "cli",
        EventSource::WebSocket => "websocket",
        EventSource::Telegram => "telegram",
        EventSource::Scheduler => "scheduler",
    }
}

fn delivery_target_label(target: &DeliveryTarget) -> &'static str {
    match target {
        DeliveryTarget::Cli => "cli",
        DeliveryTarget::File => "file",
        DeliveryTarget::WebSocket => "websocket",
        DeliveryTarget::Telegram => "telegram",
    }
}

fn render_cli_chat_line(role: &str, content: &str) -> String {
    format!("{role}> {content}")
}

fn summarize_reasoning_preview(content: &str, max_chars: usize) -> String {
    let collapsed = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        return collapsed;
    }
    let mut out = String::new();
    for ch in collapsed.chars().take(max_chars.saturating_sub(1)) {
        out.push(ch);
    }
    out.push_str("...");
    out
}

fn format_working_line(elapsed: Duration, tick: usize) -> String {
    let anim = match tick % 4 {
        0 => "",
        1 => ".",
        2 => "..",
        _ => "...",
    };
    format!("Working{anim} ({}s • esc to interrupt)", elapsed.as_secs())
}

fn reasoning_line_color_code(index: usize, total: usize) -> &'static str {
    if total <= 1 {
        return "97";
    }
    if index == 0 {
        return "90";
    }
    if index + 1 == total {
        return "97";
    }
    "37"
}

fn append_line_jsonl(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let line = serde_json::to_string(content)?;
    writeln!(file, "{line}")?;
    Ok(())
}

fn resolve_workspace_path(workspace_root: &str, raw: &str) -> PathBuf {
    let candidate = PathBuf::from(raw);
    if candidate.is_absolute() {
        return candidate;
    }
    PathBuf::from(workspace_root).join(candidate)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use anyhow::Result;

    use super::Harness;
    use crate::agent::events::{DeliveryTarget, DispatchTask, EventSource, InboundEvent};
    use crate::agent::model::{ModelBackend, ModelResponse, ModelRuntimeEvent};
    use crate::agent::permissions::PermissionPolicy;
    use crate::agent::session::SessionStore;
    use crate::agent::tools::{ToolContext, ToolRegistry, ToolSpec};
    use crate::agent::types::{Message, Role};

    struct StubModel;

    static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

    impl ModelBackend for StubModel {
        fn respond(
            &self,
            _history: &[Message],
            _tools: &[ToolSpec],
            _cancel_requested: &AtomicBool,
            _on_event: &mut dyn FnMut(ModelRuntimeEvent),
        ) -> Result<ModelResponse> {
            Ok(ModelResponse {
                output: crate::agent::model::ModelOutput::Text("ok".to_string()),
                reasoning_summaries: Vec::new(),
                usage: None,
            })
        }
    }

    fn temp_test_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let unique = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "agent_in_rust_scratch_{name}_{}_{}_{}",
            std::process::id(),
            nanos,
            unique
        ));
        fs::create_dir_all(&dir).expect("failed to create temp test directory");
        dir
    }

    fn message(index: usize) -> Message {
        Message::new(Role::User, format!("msg-{index}"))
    }

    fn harness_for_tests(compact_threshold_messages: usize) -> (Harness<StubModel>, PathBuf) {
        let dir = temp_test_dir("harness");
        let workspace_root = dir.to_string_lossy().to_string();
        let harness = Harness::new(
            StubModel,
            ToolRegistry::default(),
            PermissionPolicy::default(),
            SessionStore::new(dir.join("session.jsonl")),
            ToolContext { workspace_root },
            compact_threshold_messages,
        )
        .expect("harness should initialize");
        (harness, dir)
    }

    #[test]
    fn compact_history_keeps_last_twelve_messages() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.history = (0..20).map(message).collect();

        harness.compact_history();

        assert_eq!(harness.history.len(), 12);
        assert_eq!(
            harness.history.first().map(|m| m.content.as_str()),
            Some("msg-8")
        );
        assert_eq!(
            harness.history.last().map(|m| m.content.as_str()),
            Some("msg-19")
        );

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn compact_history_noop_when_twelve_or_fewer_messages() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.history = (0..12).map(message).collect();

        harness.compact_history();

        assert_eq!(harness.history.len(), 12);
        assert_eq!(
            harness.history.first().map(|m| m.content.as_str()),
            Some("msg-0")
        );
        assert_eq!(
            harness.history.last().map(|m| m.content.as_str()),
            Some("msg-11")
        );

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn auto_compact_triggers_over_threshold_and_trims_to_twelve() {
        let (mut harness, dir) = harness_for_tests(5);
        harness.history = (0..13).map(message).collect();

        harness.auto_compact_if_needed();

        assert_eq!(harness.history.len(), 12);
        assert_eq!(
            harness.history.first().map(|m| m.content.as_str()),
            Some("msg-1")
        );
        assert_eq!(
            harness.history.last().map(|m| m.content.as_str()),
            Some("msg-12")
        );

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn slash_compact_compacts_history() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.history = (0..15).map(message).collect();

        let handled = harness
            .handle_slash("/compact")
            .expect("slash command should succeed");

        assert!(handled, "compact slash command should be handled");
        assert_eq!(harness.history.len(), 12);
        assert_eq!(
            harness.history.first().map(|m| m.content.as_str()),
            Some("msg-3")
        );

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn inbound_cli_message_routes_to_model_and_emits_outbound_cli_event() {
        let (mut harness, dir) = harness_for_tests(60);
        harness
            .event_bus
            .publish_inbound(InboundEvent::user_message(EventSource::Cli, "hello"))
            .expect("publish inbound should succeed");

        harness
            .process_inbound_events()
            .expect("inbound processing should succeed");

        assert_eq!(
            harness
                .event_bus
                .inbound_pending_len()
                .expect("inbound count should succeed"),
            0
        );
        assert_eq!(harness.event_bus.outbound_pending_len(), 1);
        assert_eq!(harness.history.len(), 2);
        assert_eq!(harness.history[0].content, "hello");
        assert_eq!(harness.history[1].content, "ok");

        let outbound = harness
            .event_bus
            .pop_outbound()
            .expect("pop outbound should succeed")
            .expect("outbound event should exist");
        assert_eq!(outbound.target, DeliveryTarget::Cli);
        assert_eq!(outbound.content, "ok");

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn routed_user_message_handoffs_to_registered_agent() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.set_source_route(&EventSource::Cli, "support-agent");
        harness.register_agent_handler("support-agent", |harness, target, task| {
            harness.emit_assistant_output(target, format!("support handled: {}", task.content))
        });

        harness
            .event_bus
            .publish_inbound(InboundEvent::user_message(EventSource::Cli, "help me"))
            .expect("publish inbound should succeed");
        harness
            .process_inbound_events()
            .expect("inbound processing should succeed");

        let outbound = harness
            .event_bus
            .pop_outbound()
            .expect("pop outbound should succeed")
            .expect("outbound should exist");
        assert_eq!(outbound.content, "support handled: help me");
        assert_eq!(harness.history.len(), 2);
        assert_eq!(harness.history[0].content, "help me");
        assert_eq!(harness.history[1].content, "support handled: help me");

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn slash_route_updates_source_mapping() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.register_agent_handler("support-agent", |harness, target, task| {
            harness.emit_assistant_output(target, format!("support slash: {}", task.content))
        });

        let handled = harness
            .handle_slash("/route cli support-agent")
            .expect("slash route should succeed");
        assert!(handled, "route command should be handled");

        harness
            .event_bus
            .publish_inbound(InboundEvent::user_message(EventSource::Cli, "need support"))
            .expect("publish inbound should succeed");
        harness
            .process_inbound_events()
            .expect("inbound processing should succeed");

        let outbound = harness
            .event_bus
            .pop_outbound()
            .expect("pop outbound should succeed")
            .expect("outbound should exist");
        assert_eq!(outbound.content, "support slash: need support");

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn slash_dispatch_enqueues_and_processes_dispatch_task() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.register_agent_handler("delegate-agent", |harness, target, task| {
            harness.emit_assistant_output(target, format!("slash dispatch: {}", task.content))
        });

        let handled = harness
            .handle_slash("/dispatch delegate-agent run this task")
            .expect("slash dispatch should succeed");
        assert!(handled, "dispatch command should be handled");

        assert_eq!(
            harness
                .event_bus
                .inbound_pending_len()
                .expect("inbound count should succeed"),
            0
        );
        assert_eq!(harness.event_bus.outbound_pending_len(), 0);
        assert!(
            harness
                .history
                .iter()
                .any(|m| m.content == "slash dispatch: run this task"),
            "dispatch reply should be persisted to history"
        );

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn websocket_gateway_send_routes_and_stores_websocket_outbox_message() {
        let (mut harness, dir) = harness_for_tests(60);

        let handled = harness
            .handle_slash("/ws send hello from websocket")
            .expect("ws send should succeed");
        assert!(handled, "ws command should be handled");

        assert_eq!(
            harness
                .event_bus
                .inbound_pending_len()
                .expect("inbound count should succeed"),
            0
        );
        assert_eq!(harness.event_bus.outbound_pending_len(), 0);
        assert_eq!(harness.websocket_outbox.len(), 1);
        assert_eq!(harness.websocket_outbox.front(), Some(&"ok".to_string()));

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn websocket_session_route_is_sticky_after_source_route_change() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.register_agent_handler("support-agent", |harness, target, task| {
            harness.emit_assistant_output(target, format!("support: {}", task.content))
        });
        harness.set_source_route(&EventSource::WebSocket, "support-agent");

        harness
            .event_bus
            .publish_inbound(InboundEvent::websocket_message("session-a", "first"))
            .expect("publish first websocket event should succeed");
        harness
            .process_inbound_events()
            .expect("first websocket event should process");

        harness.set_source_route(&EventSource::WebSocket, "default");
        harness
            .event_bus
            .publish_inbound(InboundEvent::websocket_message("session-a", "second"))
            .expect("publish second websocket event should succeed");
        harness
            .process_inbound_events()
            .expect("second websocket event should process");

        let first = harness
            .event_bus
            .pop_outbound()
            .expect("first outbound pop should succeed")
            .expect("first outbound should exist");
        let second = harness
            .event_bus
            .pop_outbound()
            .expect("second outbound pop should succeed")
            .expect("second outbound should exist");
        assert_eq!(first.content, "support: first");
        assert_eq!(second.content, "support: second");

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn websocket_acl_denies_dispatch_to_unapproved_agent() {
        let (mut harness, dir) = harness_for_tests(60);
        harness
            .event_bus
            .upsert_websocket_session_owner("session-b", "default")
            .expect("owner setup should succeed");
        harness
            .dispatch_task(
                &EventSource::WebSocket,
                DeliveryTarget::WebSocket,
                DispatchTask::new("default", "delegate-agent", "deny me"),
                Some("session-b".to_string()),
                None,
            )
            .expect("dispatch should be processed");

        let outbound = harness
            .event_bus
            .pop_outbound()
            .expect("outbound pop should succeed")
            .expect("outbound should exist");
        assert!(
            outbound
                .content
                .contains("dispatch denied by websocket ACL")
        );

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn telegram_gateway_send_routes_and_stores_telegram_outbox_message() {
        let (mut harness, dir) = harness_for_tests(60);

        let handled = harness
            .handle_slash("/tg send 10001 hello from telegram")
            .expect("tg send should succeed");
        assert!(handled, "tg command should be handled");

        assert_eq!(
            harness
                .event_bus
                .inbound_pending_len()
                .expect("inbound count should succeed"),
            0
        );
        assert_eq!(harness.event_bus.outbound_pending_len(), 0);
        assert_eq!(harness.telegram_outbox.len(), 1);
        assert_eq!(harness.telegram_outbox.front(), Some(&"ok".to_string()));

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn telegram_chat_route_is_sticky_after_source_route_change() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.register_agent_handler("support-agent", |harness, target, task| {
            harness.emit_assistant_output(target, format!("support: {}", task.content))
        });
        harness.set_source_route(&EventSource::Telegram, "support-agent");

        harness
            .event_bus
            .publish_inbound(InboundEvent::telegram_message(22001, "first"))
            .expect("publish first telegram event should succeed");
        harness
            .process_inbound_events()
            .expect("first telegram event should process");

        harness.set_source_route(&EventSource::Telegram, "default");
        harness
            .event_bus
            .publish_inbound(InboundEvent::telegram_message(22001, "second"))
            .expect("publish second telegram event should succeed");
        harness
            .process_inbound_events()
            .expect("second telegram event should process");

        let first = harness
            .event_bus
            .pop_outbound()
            .expect("first outbound pop should succeed")
            .expect("first outbound should exist");
        let second = harness
            .event_bus
            .pop_outbound()
            .expect("second outbound pop should succeed")
            .expect("second outbound should exist");
        assert_eq!(first.content, "support: first");
        assert_eq!(second.content, "support: second");

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn telegram_acl_denies_dispatch_to_unapproved_agent() {
        let (mut harness, dir) = harness_for_tests(60);
        harness
            .event_bus
            .upsert_telegram_chat_owner(33001, "default")
            .expect("owner setup should succeed");
        harness
            .dispatch_task(
                &EventSource::Telegram,
                DeliveryTarget::Telegram,
                DispatchTask::new("default", "delegate-agent", "deny me"),
                None,
                Some(33001),
            )
            .expect("dispatch should be processed");

        let outbound = harness
            .event_bus
            .pop_outbound()
            .expect("outbound pop should succeed")
            .expect("outbound should exist");
        assert!(outbound.content.contains("dispatch denied by telegram ACL"));

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn flush_outbound_file_target_writes_jsonl() {
        let (mut harness, dir) = harness_for_tests(60);
        harness
            .emit_assistant_output(DeliveryTarget::File, "persist me".to_string())
            .expect("emit should succeed");
        harness
            .flush_outbound_events()
            .expect("flush outbound should succeed");

        let written = fs::read_to_string(dir.join(".agent").join("outbox.jsonl"))
            .expect("outbox should be readable");
        assert!(written.contains("\"persist me\""));
        assert_eq!(harness.event_bus.outbound_pending_len(), 0);

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn delivery_failure_increments_attempt_count_and_preserves_pending_event() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.file_outbox_path = dir.join("missing").join("fail.jsonl");
        fs::create_dir_all(dir.join("missing")).expect("parent directory should be created");
        fs::write(&harness.file_outbox_path, "existing").expect("seed outbox file");
        let mut perms = fs::metadata(&harness.file_outbox_path)
            .expect("metadata should load")
            .permissions();
        perms.set_readonly(true);
        fs::set_permissions(&harness.file_outbox_path, perms).expect("permissions should set");

        harness
            .emit_assistant_output(DeliveryTarget::File, "should fail".to_string())
            .expect("emit should succeed");
        let pending = harness
            .event_bus
            .next_outbound_for_delivery()
            .expect("lookup pending should succeed")
            .expect("pending outbound should exist");

        harness
            .flush_outbound_events()
            .expect("flush should handle delivery failure");

        assert_eq!(harness.event_bus.outbound_pending_len(), 1);
        assert_eq!(harness.event_bus.outbound_attempt_count(pending.id), 1);

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn inbound_processing_respects_max_events_per_cycle() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.max_inbound_events_per_cycle = 1;

        harness
            .event_bus
            .publish_inbound(InboundEvent::user_message(EventSource::Cli, "one"))
            .expect("publish one should succeed");
        harness
            .event_bus
            .publish_inbound(InboundEvent::user_message(EventSource::Cli, "two"))
            .expect("publish two should succeed");

        harness
            .process_inbound_events()
            .expect("first cycle should succeed");
        assert_eq!(
            harness
                .event_bus
                .inbound_pending_len()
                .expect("inbound count should succeed"),
            1
        );

        harness
            .process_inbound_events()
            .expect("second cycle should succeed");
        assert_eq!(
            harness
                .event_bus
                .inbound_pending_len()
                .expect("inbound count should succeed"),
            0
        );

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn outbound_flush_respects_max_events_per_cycle() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.max_outbound_events_per_cycle = 1;
        harness
            .emit_assistant_output(DeliveryTarget::WebSocket, "first".to_string())
            .expect("emit first should succeed");
        harness
            .emit_assistant_output(DeliveryTarget::WebSocket, "second".to_string())
            .expect("emit second should succeed");

        harness
            .flush_outbound_events()
            .expect("first flush should succeed");
        assert_eq!(harness.event_bus.outbound_pending_len(), 1);
        assert_eq!(harness.websocket_outbox.len(), 1);

        harness
            .flush_outbound_events()
            .expect("second flush should succeed");
        assert_eq!(harness.event_bus.outbound_pending_len(), 0);
        assert_eq!(harness.websocket_outbox.len(), 2);

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn explicit_dispatch_event_runs_target_agent_handler() {
        let (mut harness, dir) = harness_for_tests(60);
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen_clone = Arc::clone(&seen);
        harness.register_agent_handler("delegate-agent", move |harness, target, task| {
            seen_clone
                .lock()
                .expect("lock should succeed")
                .push(format!("{}->{}", task.from_agent, task.to_agent));
            harness.emit_assistant_output(target, format!("delegate reply: {}", task.content))
        });

        let dispatch = DispatchTask::new("default", "delegate-agent", "take this");
        harness
            .event_bus
            .publish_inbound(InboundEvent::dispatch_task(EventSource::Cli, dispatch))
            .expect("publish dispatch should succeed");
        harness
            .process_inbound_events()
            .expect("inbound processing should succeed");

        let traces = seen.lock().expect("lock should succeed");
        assert_eq!(traces.as_slice(), ["default->delegate-agent"]);
        drop(traces);

        let outbound = harness
            .event_bus
            .pop_outbound()
            .expect("pop outbound should succeed")
            .expect("outbound should exist");
        assert_eq!(outbound.content, "delegate reply: take this");

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn failed_dispatch_does_not_block_follow_up_events() {
        let (mut harness, dir) = harness_for_tests(60);
        harness.register_agent_handler("broken-agent", |_harness, _target, _task| {
            anyhow::bail!("simulated handler failure")
        });
        harness.register_agent_handler("healthy-agent", |harness, target, task| {
            harness.emit_assistant_output(target, format!("healthy: {}", task.content))
        });

        harness
            .event_bus
            .publish_inbound(InboundEvent::dispatch_task(
                EventSource::Cli,
                DispatchTask::new("default", "broken-agent", "first"),
            ))
            .expect("publish broken dispatch should succeed");
        harness
            .event_bus
            .publish_inbound(InboundEvent::dispatch_task(
                EventSource::Cli,
                DispatchTask::new("default", "healthy-agent", "second"),
            ))
            .expect("publish healthy dispatch should succeed");

        harness
            .process_inbound_events()
            .expect("inbound processing should continue after failure");

        let first = harness
            .event_bus
            .pop_outbound()
            .expect("pop outbound should succeed")
            .expect("first outbound should exist");
        let second = harness
            .event_bus
            .pop_outbound()
            .expect("pop outbound should succeed")
            .expect("second outbound should exist");
        assert!(first.content.contains("event processing error"));
        assert_eq!(second.content, "healthy: second");

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn slash_plan_creates_artifacts_without_error() {
        let (mut harness, dir) = harness_for_tests(60);

        let handled = harness
            .handle_slash("/plan add planner and evaluator")
            .expect("plan command should succeed");
        assert!(handled);

        let artifacts_dir = dir.join(".agent").join("artifacts");
        assert!(artifacts_dir.exists());

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn slash_post_websocket_enqueues_outbound() {
        let (mut harness, dir) = harness_for_tests(60);

        let handled = harness
            .handle_slash("/post websocket from post")
            .expect("post command should succeed");
        assert!(handled);

        assert!(harness.websocket_outbox.iter().any(|m| m == "from post"));

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn dispatch_policy_scope_can_block_specific_source_and_agent() {
        let dir = temp_test_dir("dispatch_policy_scope");
        let workspace_root = dir.to_string_lossy().to_string();
        fs::create_dir_all(dir.join(".agent")).expect("agent dir should exist");
        fs::write(
            dir.join(".agent").join("policy.json"),
            r#"{
  "default_effect": "allow",
  "rules": [
    {
      "effect": "deny",
      "action": "^dispatch:default$",
      "action_matcher": "regex",
      "scope": {
        "source": "websocket",
        "agent": "default"
      }
    }
  ]
}"#,
        )
        .expect("policy file write should succeed");

        let mut harness = Harness::new(
            StubModel,
            ToolRegistry::default(),
            PermissionPolicy::default(),
            SessionStore::new(dir.join("session.jsonl")),
            ToolContext { workspace_root },
            60,
        )
        .expect("harness should initialize");

        harness
            .event_bus
            .publish_inbound(InboundEvent::user_message(
                EventSource::WebSocket,
                "deny this",
            ))
            .expect("publish should succeed");
        harness
            .process_inbound_events()
            .expect("processing should succeed");

        let outbound = harness
            .event_bus
            .pop_outbound()
            .expect("pop should succeed")
            .expect("outbound should exist");
        assert!(outbound.content.contains("dispatch denied by policy"));

        drop(harness);
        fs::remove_dir_all(dir).expect("failed to clean temp directory");
    }

    #[test]
    fn render_cli_chat_line_formats_user_role() {
        let rendered = super::render_cli_chat_line("user", "hello");
        assert_eq!(rendered, "user> hello");
    }

    #[test]
    fn render_cli_chat_line_formats_assistant_role() {
        let rendered = super::render_cli_chat_line("assistant", "how can I help?");
        assert_eq!(rendered, "assistant> how can I help?");
    }

    #[test]
    fn reasoning_overlay_keeps_latest_lines_with_configured_capacity() {
        let mut overlay = super::CliReasoningOverlay {
            enabled: true,
            is_tty: false,
            max_lines: 3,
            rendered_lines: 0,
            lines: std::collections::VecDeque::new(),
            working_elapsed: None,
            working_tick: 0,
        };
        overlay
            .push("one")
            .expect("push should succeed when tty rendering is disabled");
        overlay
            .push("two")
            .expect("push should succeed when tty rendering is disabled");
        overlay
            .push("three")
            .expect("push should succeed when tty rendering is disabled");
        overlay
            .push("four")
            .expect("push should succeed when tty rendering is disabled");

        let lines: Vec<&str> = overlay.lines.iter().map(|s| s.as_str()).collect();
        assert_eq!(lines, vec!["two", "three", "four"]);
    }

    #[test]
    fn reasoning_line_color_code_fades_top_and_highlights_bottom() {
        assert_eq!(super::reasoning_line_color_code(0, 4), "90");
        assert_eq!(super::reasoning_line_color_code(1, 4), "37");
        assert_eq!(super::reasoning_line_color_code(3, 4), "97");
    }

    #[test]
    fn format_working_line_reports_seconds_and_animation() {
        let line0 = super::format_working_line(Duration::from_secs(57), 0);
        let line3 = super::format_working_line(Duration::from_secs(57), 3);

        assert_eq!(line0, "Working (57s • esc to interrupt)");
        assert_eq!(line3, "Working... (57s • esc to interrupt)");
    }
}
