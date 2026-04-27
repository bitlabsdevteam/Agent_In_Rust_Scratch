use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::tools::ToolSpec;
use crate::agent::types::{Message, Role};

#[derive(Debug, Clone)]
pub enum ModelOutput {
    Text(String),
    ToolCalls(Vec<ModelToolCall>),
}

#[derive(Debug, Clone)]
pub struct ModelToolCall {
    pub name: String,
    pub args: String,
}

#[derive(Debug, Clone, Default)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct ModelResponse {
    pub output: ModelOutput,
    pub reasoning_summaries: Vec<String>,
    pub usage: Option<ModelUsage>,
}

impl ModelResponse {
    fn from_output(output: ModelOutput) -> Self {
        Self {
            output,
            reasoning_summaries: Vec::new(),
            usage: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ModelRuntimeEvent {
    ReasoningSummaryDelta(String),
    OutputTextDelta(String),
}

pub trait ModelBackend {
    fn respond(
        &self,
        history: &[Message],
        tools: &[ToolSpec],
        cancel_requested: &AtomicBool,
        on_event: &mut dyn FnMut(ModelRuntimeEvent),
    ) -> Result<ModelResponse>;
}

pub enum AnyModel {
    OpenAI(OpenAIChatModel),
    Anthropic(AnthropicChatModel),
    Local(LocalModel),
}

impl AnyModel {
    pub fn from_env() -> Result<Self> {
        let provider = env::var("MODEL_PROVIDER").unwrap_or_else(|_| "openai".to_string());
        match provider.as_str() {
            "openai" => Ok(Self::OpenAI(OpenAIChatModel::from_env()?)),
            "anthropic" => Ok(Self::Anthropic(AnthropicChatModel::from_env()?)),
            "local" => Ok(Self::Local(LocalModel::from_env()?)),
            other => {
                bail!("unsupported MODEL_PROVIDER `{other}` (supported: openai, anthropic, local)")
            }
        }
    }
}

impl ModelBackend for AnyModel {
    fn respond(
        &self,
        history: &[Message],
        tools: &[ToolSpec],
        cancel_requested: &AtomicBool,
        on_event: &mut dyn FnMut(ModelRuntimeEvent),
    ) -> Result<ModelResponse> {
        match self {
            Self::OpenAI(m) => m.respond(history, tools, cancel_requested, on_event),
            Self::Anthropic(m) => m.respond(history, tools, cancel_requested, on_event),
            Self::Local(m) => m.respond(history, tools, cancel_requested, on_event),
        }
    }
}

pub struct OpenAIChatModel {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    system_prompt: String,
    reasoning_summary: Option<String>,
}

impl OpenAIChatModel {
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("OPENAI_API_KEY")
            .context("OPENAI_API_KEY is missing. Set it in your .env file.")?;
        if api_key.trim().is_empty() {
            bail!("OPENAI_API_KEY is empty.");
        }

        let model = env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_string());
        let base_url =
            env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
        let system_prompt = env::var("OPENAI_SYSTEM_PROMPT").unwrap_or_else(|_| {
            "You are a pragmatic coding assistant operating in a local terminal harness. Return tool calls via tool schema when required."
                .to_string()
        });
        let reasoning_summary = match env::var("OPENAI_REASONING_SUMMARY") {
            Ok(raw) => {
                let value = raw.trim();
                if value.is_empty() || value.eq_ignore_ascii_case("off") {
                    None
                } else {
                    Some(value.to_string())
                }
            }
            Err(_) => None,
        };

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
            base_url,
            system_prompt,
            reasoning_summary,
        })
    }
}

impl ModelBackend for OpenAIChatModel {
    fn respond(
        &self,
        history: &[Message],
        tools: &[ToolSpec],
        cancel_requested: &AtomicBool,
        on_event: &mut dyn FnMut(ModelRuntimeEvent),
    ) -> Result<ModelResponse> {
        let mut request_messages = vec![ResponseInputMessage::new(
            "system",
            "input_text",
            self.system_prompt.clone(),
        )];

        for msg in history {
            let role = map_input_role(&msg.role);
            let content_type = map_input_content_type(&msg.role);
            request_messages.push(ResponseInputMessage::new(
                role,
                content_type,
                msg.content.clone(),
            ));
        }

        let tool_defs: Vec<ResponseToolDefinition> = tools
            .iter()
            .map(|tool| ResponseToolDefinition {
                ty: "function".to_string(),
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "string",
                            "description": "Tool arguments serialized as one plain text string."
                        }
                    },
                    "required": ["args"],
                    "additionalProperties": false
                }),
                strict: true,
            })
            .collect();

        let body = ResponseApiRequest {
            model: self.model.clone(),
            input: request_messages,
            stream: true,
            tools: if tool_defs.is_empty() {
                None
            } else {
                Some(tool_defs)
            },
            reasoning: self
                .reasoning_summary
                .as_ref()
                .map(|summary| ResponseReasoning {
                    summary: summary.clone(),
                }),
        };

        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .context("failed to call OpenAI Responses API")?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().unwrap_or_else(|_| "".to_string());
            bail!("OpenAI API error {}: {}", status.as_u16(), text);
        }

        let mut final_response: Option<Value> = None;
        let mut final_reasoning_parts: BTreeMap<usize, String> = BTreeMap::new();
        let mut summary_part_buffers: BTreeMap<usize, String> = BTreeMap::new();
        let mut stream_error: Option<String> = None;

        let mut reader = BufReader::new(response);
        let mut line = String::new();
        let mut data_lines: Vec<String> = Vec::new();

        loop {
            if cancel_requested.load(Ordering::Relaxed) {
                bail!("model call canceled by user");
            }
            line.clear();
            let read = reader.read_line(&mut line)?;
            if read == 0 {
                if !data_lines.is_empty() {
                    let payload = data_lines.join("\n");
                    handle_responses_stream_data(
                        payload.as_str(),
                        &mut final_response,
                        &mut final_reasoning_parts,
                        &mut summary_part_buffers,
                        &mut stream_error,
                        on_event,
                    )?;
                }
                break;
            }

            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                if !data_lines.is_empty() {
                    let payload = data_lines.join("\n");
                    handle_responses_stream_data(
                        payload.as_str(),
                        &mut final_response,
                        &mut final_reasoning_parts,
                        &mut summary_part_buffers,
                        &mut stream_error,
                        on_event,
                    )?;
                    data_lines.clear();
                }
                continue;
            }

            if let Some(data) = trimmed.strip_prefix("data:") {
                data_lines.push(data.trim_start().to_string());
            }
        }

        if let Some(err) = stream_error {
            bail!("OpenAI Responses API stream failed: {err}");
        }

        let response_obj =
            final_response.context("OpenAI stream ended without response.completed")?;
        let output = parse_responses_model_output(&response_obj)?;
        let usage = parse_responses_usage(&response_obj);

        let mut reasoning_summaries: Vec<String> = final_reasoning_parts
            .into_values()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if reasoning_summaries.is_empty() {
            reasoning_summaries = collect_reasoning_summaries(&response_obj);
        }

        Ok(ModelResponse {
            output,
            reasoning_summaries,
            usage,
        })
    }
}

fn handle_responses_stream_data(
    payload: &str,
    final_response: &mut Option<Value>,
    final_reasoning_parts: &mut BTreeMap<usize, String>,
    summary_part_buffers: &mut BTreeMap<usize, String>,
    stream_error: &mut Option<String>,
    on_event: &mut dyn FnMut(ModelRuntimeEvent),
) -> Result<()> {
    if payload == "[DONE]" {
        return Ok(());
    }

    let event: Value = serde_json::from_str(payload)
        .with_context(|| format!("failed to parse Responses stream event: {payload}"))?;
    let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");

    match event_type {
        "response.completed" => {
            *final_response = event.get("response").cloned();
        }
        "response.failed" => {
            let msg = event
                .get("response")
                .and_then(|r| r.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("unknown failure")
                .to_string();
            *stream_error = Some(msg);
        }
        "response.reasoning_summary_text.delta" => {
            if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                let summary_index = event
                    .get("summary_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                let entry = summary_part_buffers.entry(summary_index).or_default();
                entry.push_str(delta);
                on_event(ModelRuntimeEvent::ReasoningSummaryDelta(entry.clone()));
            }
        }
        "response.reasoning_summary.delta" => {
            if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                let summary_index = event
                    .get("summary_index")
                    .or_else(|| event.get("index"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                let entry = summary_part_buffers.entry(summary_index).or_default();
                entry.push_str(delta);
                on_event(ModelRuntimeEvent::ReasoningSummaryDelta(entry.clone()));
            }
        }
        "response.reasoning_summary_part.done" => {
            if let Some(text) = event
                .get("part")
                .and_then(|p| p.get("text"))
                .and_then(Value::as_str)
            {
                let summary_index = event
                    .get("summary_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                final_reasoning_parts.insert(summary_index, text.to_string());
                summary_part_buffers.remove(&summary_index);
            }
        }
        "response.reasoning_summary.done" => {
            if let Some(text) = event
                .get("summary")
                .or_else(|| event.get("text"))
                .and_then(Value::as_str)
            {
                let summary_index = event
                    .get("summary_index")
                    .or_else(|| event.get("index"))
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                final_reasoning_parts.insert(summary_index, text.to_string());
                summary_part_buffers.remove(&summary_index);
            }
        }
        "response.reasoning_summary_text.done" => {
            if let Some(text) = event.get("text").and_then(Value::as_str) {
                let summary_index = event
                    .get("summary_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                final_reasoning_parts.insert(summary_index, text.to_string());
                summary_part_buffers.remove(&summary_index);
            }
        }
        "response.output_text.delta" => {
            if let Some(delta) = event.get("delta").and_then(Value::as_str) {
                on_event(ModelRuntimeEvent::OutputTextDelta(delta.to_string()));
            }
        }
        _ => {}
    }

    Ok(())
}

fn parse_responses_model_output(response: &Value) -> Result<ModelOutput> {
    let mut tool_calls = Vec::new();
    if let Some(items) = response.get("output").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("function_call") {
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let args = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .map(parse_args_field)
                    .unwrap_or_default();
                if !name.is_empty() {
                    tool_calls.push(ModelToolCall { name, args });
                }
            }
        }
    }

    if !tool_calls.is_empty() {
        return Ok(ModelOutput::ToolCalls(tool_calls));
    }

    let text = response
        .get("output_text")
        .and_then(Value::as_str)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| collect_response_message_text(response));

    parse_protocol_or_text(&text)
}

fn collect_response_message_text(response: &Value) -> String {
    let mut chunks = Vec::new();
    if let Some(items) = response.get("output").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) != Some("message") {
                continue;
            }
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for part in content {
                    if part.get("type").and_then(Value::as_str) == Some("output_text") {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            chunks.push(text.to_string());
                        }
                    }
                }
            }
        }
    }
    chunks.join("\n")
}

fn collect_reasoning_summaries(response: &Value) -> Vec<String> {
    let mut summaries = Vec::new();
    if let Some(items) = response.get("output").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) != Some("reasoning") {
                continue;
            }
            if let Some(text) = item.get("summary").and_then(Value::as_str) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    summaries.push(trimmed.to_string());
                }
                continue;
            }
            if let Some(parts) = item.get("summary").and_then(Value::as_array) {
                for part in parts {
                    if let Some(text) = part
                        .get("text")
                        .or_else(|| part.get("summary_text"))
                        .and_then(Value::as_str)
                    {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            summaries.push(trimmed.to_string());
                        }
                    }
                }
            }
        }
    }
    summaries
}

fn parse_responses_usage(response: &Value) -> Option<ModelUsage> {
    let usage = response.get("usage")?;
    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let reasoning_tokens = usage
        .get("reasoning_tokens")
        .or_else(|| {
            usage
                .get("output_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
        })
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    Some(ModelUsage {
        input_tokens,
        output_tokens,
        reasoning_tokens,
        total_tokens,
    })
}

pub struct AnthropicChatModel {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    system_prompt: String,
}

impl AnthropicChatModel {
    pub fn from_env() -> Result<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .context("ANTHROPIC_API_KEY is missing. Set it in your .env file.")?;
        if api_key.trim().is_empty() {
            bail!("ANTHROPIC_API_KEY is empty.");
        }

        let model =
            env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-3-5-sonnet-latest".to_string());
        let base_url = env::var("ANTHROPIC_BASE_URL")
            .unwrap_or_else(|_| "https://api.anthropic.com/v1".to_string());
        let system_prompt = env::var("ANTHROPIC_SYSTEM_PROMPT").unwrap_or_else(|_| {
            "You are a pragmatic coding assistant. Use JSON protocol responses when possible."
                .to_string()
        });

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
            base_url,
            system_prompt,
        })
    }
}

impl ModelBackend for AnthropicChatModel {
    fn respond(
        &self,
        history: &[Message],
        _tools: &[ToolSpec],
        _cancel_requested: &AtomicBool,
        _on_event: &mut dyn FnMut(ModelRuntimeEvent),
    ) -> Result<ModelResponse> {
        let mut messages = Vec::new();
        for msg in history {
            let role = match msg.role {
                Role::User => "user",
                _ => "assistant",
            };
            messages.push(json!({
                "role": role,
                "content": msg.content,
            }));
        }

        let body = json!({
            "model": self.model,
            "system": self.system_prompt,
            "max_tokens": 800,
            "messages": messages,
        });

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .context("failed to call Anthropic API")?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().unwrap_or_else(|_| "".to_string());
            bail!("Anthropic API error {}: {}", status.as_u16(), text);
        }

        let parsed: AnthropicResponse = response
            .json()
            .context("failed to parse Anthropic response")?;
        let text = parsed
            .content
            .iter()
            .find(|block| block.r#type == "text")
            .map(|block| block.text.clone())
            .unwrap_or_default();

        Ok(ModelResponse::from_output(parse_protocol_or_text(&text)?))
    }
}

pub struct LocalModel {
    system_prompt: String,
}

impl LocalModel {
    pub fn from_env() -> Result<Self> {
        let system_prompt = env::var("LOCAL_SYSTEM_PROMPT")
            .unwrap_or_else(|_| "You are a local fallback model.".to_string());
        Ok(Self { system_prompt })
    }
}

impl ModelBackend for LocalModel {
    fn respond(
        &self,
        history: &[Message],
        _tools: &[ToolSpec],
        _cancel_requested: &AtomicBool,
        _on_event: &mut dyn FnMut(ModelRuntimeEvent),
    ) -> Result<ModelResponse> {
        let last = history
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::User))
            .map(|m| m.content.trim().to_string())
            .unwrap_or_else(|| "".to_string());

        if let Some(rest) = last.strip_prefix("tool:") {
            let mut parts = rest.splitn(2, char::is_whitespace);
            let name = parts.next().unwrap_or("").trim();
            let args = parts.next().unwrap_or("").trim();
            if !name.is_empty() {
                return Ok(ModelResponse::from_output(ModelOutput::ToolCalls(vec![
                    ModelToolCall {
                        name: name.to_string(),
                        args: args.to_string(),
                    },
                ])));
            }
        }

        let text = if last.is_empty() {
            self.system_prompt.clone()
        } else {
            format!("local model reply: {last}")
        };
        Ok(ModelResponse::from_output(ModelOutput::Text(text)))
    }
}

fn parse_protocol_or_text(content: &str) -> Result<ModelOutput> {
    if let Some(output) = parse_structured_protocol(content)? {
        return Ok(output);
    }

    if content.trim().is_empty() {
        bail!("model returned an empty response.");
    }
    Ok(ModelOutput::Text(content.to_string()))
}

fn parse_structured_protocol(content: &str) -> Result<Option<ModelOutput>> {
    let parsed: Value = match serde_json::from_str(content) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };

    let ty = parsed.get("type").and_then(Value::as_str).unwrap_or("");
    match ty {
        "text" => {
            let text = parsed
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            Ok(Some(ModelOutput::Text(text)))
        }
        "tool_calls" => {
            let calls = parsed
                .get("calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut tool_calls = Vec::new();
            for call in calls {
                let name = call
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let args = call
                    .get("args")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if !name.is_empty() {
                    tool_calls.push(ModelToolCall { name, args });
                }
            }
            Ok(Some(ModelOutput::ToolCalls(tool_calls)))
        }
        _ => Ok(None),
    }
}

fn parse_args_field(arguments: &str) -> String {
    let parsed: Result<Value, _> = serde_json::from_str(arguments);
    if let Ok(v) = parsed {
        if let Some(args) = v.get("args").and_then(Value::as_str) {
            return args.to_string();
        }
    }
    arguments.to_string()
}

fn map_input_role(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "assistant",
    }
}

fn map_input_content_type(role: &Role) -> &'static str {
    match role {
        Role::Assistant | Role::Tool => "output_text",
        Role::System | Role::User => "input_text",
    }
}

#[derive(Debug, Serialize)]
struct ResponseApiRequest {
    model: String,
    input: Vec<ResponseInputMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ResponseToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ResponseReasoning>,
}

#[derive(Debug, Serialize)]
struct ResponseInputMessage {
    #[serde(rename = "type")]
    ty: String,
    role: String,
    content: Vec<ResponseInputText>,
}

impl ResponseInputMessage {
    fn new(role: &str, content_type: &str, text: String) -> Self {
        Self {
            ty: "message".to_string(),
            role: role.to_string(),
            content: vec![ResponseInputText {
                ty: content_type.to_string(),
                text,
            }],
        }
    }
}

#[derive(Debug, Serialize)]
struct ResponseInputText {
    #[serde(rename = "type")]
    ty: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct ResponseToolDefinition {
    #[serde(rename = "type")]
    ty: String,
    name: String,
    description: String,
    parameters: Value,
    strict: bool,
}

#[derive(Debug, Serialize)]
struct ResponseReasoning {
    summary: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    r#type: String,
    #[serde(default)]
    text: String,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        ModelOutput, collect_reasoning_summaries, parse_responses_usage, parse_structured_protocol,
    };

    #[test]
    fn parses_structured_text_protocol() {
        let payload = r#"{"type":"text","text":"hello"}"#;
        let out = parse_structured_protocol(payload)
            .expect("parse should succeed")
            .expect("protocol should be recognized");

        match out {
            ModelOutput::Text(text) => assert_eq!(text, "hello"),
            _ => panic!("expected text output"),
        }
    }

    #[test]
    fn parses_structured_tool_calls_protocol() {
        let payload = r#"{"type":"tool_calls","calls":[{"name":"echo","args":"hi"}]}"#;
        let out = parse_structured_protocol(payload)
            .expect("parse should succeed")
            .expect("protocol should be recognized");

        match out {
            ModelOutput::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "echo");
                assert_eq!(calls[0].args, "hi");
            }
            _ => panic!("expected tool calls output"),
        }
    }

    #[test]
    fn collects_reasoning_summary_texts_from_response_output() {
        let response = json!({
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "first"},
                        {"type": "summary_text", "text": "second"}
                    ]
                }
            ]
        });

        let summaries = collect_reasoning_summaries(&response);
        assert_eq!(summaries, vec!["first".to_string(), "second".to_string()]);
    }

    #[test]
    fn parses_reasoning_tokens_from_usage_payload() {
        let response = json!({
            "usage": {
                "input_tokens": 11,
                "output_tokens": 22,
                "output_tokens_details": {
                    "reasoning_tokens": 7
                },
                "total_tokens": 33
            }
        });

        let usage = parse_responses_usage(&response).expect("usage should parse");
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 22);
        assert_eq!(usage.reasoning_tokens, 7);
        assert_eq!(usage.total_tokens, 33);
    }
}
