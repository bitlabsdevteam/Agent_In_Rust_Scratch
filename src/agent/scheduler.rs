use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;

use crate::agent::events::{DispatchTask, EventBus, EventSource, InboundEvent, InboundPayload};

#[derive(Clone)]
pub struct SchedulerControl {
    enabled: Arc<AtomicBool>,
    heartbeat_interval_secs: Arc<AtomicU64>,
    max_pending_inbound: Arc<AtomicUsize>,
}

impl SchedulerControl {
    pub fn new(enabled: bool, heartbeat_interval_secs: u64, max_pending_inbound: usize) -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(enabled)),
            heartbeat_interval_secs: Arc::new(AtomicU64::new(heartbeat_interval_secs.max(1))),
            max_pending_inbound: Arc::new(AtomicUsize::new(max_pending_inbound.max(1))),
        }
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    pub fn set_heartbeat_interval_secs(&self, secs: u64) {
        self.heartbeat_interval_secs
            .store(secs.max(1), Ordering::Relaxed);
    }

    pub fn set_max_pending_inbound(&self, max_pending_inbound: usize) {
        self.max_pending_inbound
            .store(max_pending_inbound.max(1), Ordering::Relaxed);
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn heartbeat_interval(&self) -> Duration {
        Duration::from_secs(self.heartbeat_interval_secs.load(Ordering::Relaxed).max(1))
    }

    fn max_pending_inbound(&self) -> usize {
        self.max_pending_inbound.load(Ordering::Relaxed).max(1)
    }
}

pub struct SchedulerDaemon {
    stop_tx: Option<Sender<()>>,
    join_handle: Option<JoinHandle<()>>,
}

impl SchedulerDaemon {
    pub fn start(event_db_path: PathBuf, control: SchedulerControl) -> Self {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let join_handle = thread::spawn(move || {
            let mut bus = match EventBus::open(&event_db_path) {
                Ok(bus) => bus,
                Err(_) => return,
            };

            loop {
                let timeout = control.heartbeat_interval();
                if stop_rx.recv_timeout(timeout).is_ok() {
                    break;
                }

                if !control.enabled() {
                    continue;
                }

                let pending = match bus.inbound_pending_len() {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if pending >= control.max_pending_inbound() {
                    continue;
                }

                let _ = enqueue_heartbeat(&mut bus);
            }
        });

        Self {
            stop_tx: Some(stop_tx),
            join_handle: Some(join_handle),
        }
    }

    pub fn stop(&mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SchedulerDaemon {
    fn drop(&mut self) {
        self.stop();
    }
}

pub fn enqueue_heartbeat(bus: &mut EventBus) -> Result<()> {
    bus.publish_inbound(InboundEvent {
        source: EventSource::Scheduler,
        payload: InboundPayload::DispatchTask(DispatchTask::new(
            "scheduler",
            "default",
            "__heartbeat__",
        )),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use anyhow::Result;

    use super::{SchedulerControl, SchedulerDaemon, enqueue_heartbeat};
    use crate::agent::events::EventBus;

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
    fn enqueue_heartbeat_publishes_dispatch_event() -> Result<()> {
        let dir = temp_test_dir("scheduler_enqueue");
        let mut bus = EventBus::open(dir.join("events.db"))?;

        enqueue_heartbeat(&mut bus)?;

        assert_eq!(bus.inbound_pending_len()?, 1);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }

    #[test]
    fn daemon_respects_backpressure_limit() -> Result<()> {
        let dir = temp_test_dir("scheduler_backpressure");
        let db = dir.join("events.db");

        {
            let mut bus = EventBus::open(&db)?;
            enqueue_heartbeat(&mut bus)?;
        }

        let control = SchedulerControl::new(true, 1, 1);
        let mut daemon = SchedulerDaemon::start(db.clone(), control);
        std::thread::sleep(Duration::from_millis(1100));
        daemon.stop();

        let bus = EventBus::open(&db)?;
        assert_eq!(bus.inbound_pending_len()?, 1);

        fs::remove_dir_all(dir).expect("failed to cleanup temp directory");
        Ok(())
    }
}
