use std::collections::HashMap;

use crate::agent::events::EventSource;

#[derive(Debug, Clone)]
pub struct SourceRouter {
    default_agent: String,
    routes: HashMap<String, String>,
}

impl Default for SourceRouter {
    fn default() -> Self {
        Self {
            default_agent: "default".to_string(),
            routes: HashMap::new(),
        }
    }
}

impl SourceRouter {
    pub fn set_route(&mut self, source: &EventSource, agent_id: impl Into<String>) {
        self.routes
            .insert(source.key().to_string(), agent_id.into());
    }

    pub fn route(&self, source: &EventSource) -> &str {
        self.routes
            .get(source.key())
            .map(String::as_str)
            .unwrap_or(self.default_agent.as_str())
    }
}

trait SourceKey {
    fn key(&self) -> &'static str;
}

impl SourceKey for EventSource {
    fn key(&self) -> &'static str {
        match self {
            EventSource::Cli => "cli",
            EventSource::WebSocket => "websocket",
            EventSource::Scheduler => "scheduler",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SourceRouter;
    use crate::agent::events::EventSource;

    #[test]
    fn router_defaults_to_default_agent_for_unknown_source() {
        let router = SourceRouter::default();
        assert_eq!(router.route(&EventSource::Cli), "default");
    }

    #[test]
    fn router_uses_explicit_route_for_source() {
        let mut router = SourceRouter::default();
        let source = EventSource::Cli;
        router.set_route(&source, "support-agent");
        assert_eq!(router.route(&source), "support-agent");
    }
}
