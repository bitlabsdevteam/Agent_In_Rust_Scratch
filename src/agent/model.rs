use std::env;

use anyhow::{Context, Result, bail};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::tools::ToolSpec;
use crate::agent::types::{Message, Role};

pub enum ModelOutput {
    Text(String),
    ToolCalls(Vec<ModelToolCall>),
}

#[derive(Debug, Clone)]
pub struct ModelToolCall {
    pub name: String,
    pub args: String,
}

pub trait ModelBackend {
    fn respond(&self, history: &[Message], tools: &[ToolSpec]) -> Result<ModelOutput>;
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
    fn respond(&self, history: &[Message], tools: &[ToolSpec]) -> Result<ModelOutput> {
        match self {
            Self::OpenAI(m) => m.respond(history, tools),
            Self::Anthropic(m) => m.respond(history, tools),
            Self::Local(m) => m.respond(history, tools),
        }
    }
}

pub struct OpenAIChatModel {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    system_prompt: String,
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

        Ok(Self {
            client: Client::new(),
            api_key,
            model,
            base_url,
            system_prompt,
        })
    }
}

impl ModelBackend for OpenAIChatModel {
    fn respond(&self, history: &[Message], tools: &[ToolSpec]) -> Result<ModelOutput> {
        let mut request_messages = vec![ChatMessage {
            role: "system".to_string(),
            content: Some(self.system_prompt.clone()),
        }];

        for msg in history {
            request_messages.push(ChatMessage {
                role: map_role(&msg.role).to_string(),
                content: Some(msg.content.clone()),
            });
        }

        let tool_defs: Vec<ToolDefinition> = tools
            .iter()
            .map(|tool| ToolDefinition {
                r#type: "function".to_string(),
                function: ToolFunction {
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
                },
            })
            .collect();

        let body = ChatCompletionRequest {
            model: self.model.clone(),
            messages: request_messages,
            temperature: 0.2,
            tools: if tool_defs.is_empty() {
                None
            } else {
                Some(tool_defs)
            },
        };

        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let response = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .context("failed to call OpenAI API")?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().unwrap_or_else(|_| "".to_string());
            bail!("OpenAI API error {}: {}", status.as_u16(), text);
        }

        let parsed: ChatCompletionResponse = response
            .json()
            .context("failed to parse OpenAI chat completion response")?;
        let message = parsed
            .choices
            .first()
            .map(|c| &c.message)
            .context("OpenAI returned no choices")?;

        if let Some(calls) = &message.tool_calls {
            let parsed_calls: Vec<ModelToolCall> = calls
                .iter()
                .map(|call| ModelToolCall {
                    name: call.function.name.clone(),
                    args: parse_args_field(&call.function.arguments),
                })
                .collect();

            if !parsed_calls.is_empty() {
                return Ok(ModelOutput::ToolCalls(parsed_calls));
            }
        }

        let content = message.content.clone().unwrap_or_else(|| "".to_string());
        parse_protocol_or_text(&content)
    }
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
    fn respond(&self, history: &[Message], _tools: &[ToolSpec]) -> Result<ModelOutput> {
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

        parse_protocol_or_text(&text)
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
    fn respond(&self, history: &[Message], _tools: &[ToolSpec]) -> Result<ModelOutput> {
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
                return Ok(ModelOutput::ToolCalls(vec![ModelToolCall {
                    name: name.to_string(),
                    args: args.to_string(),
                }]));
            }
        }

        let text = if last.is_empty() {
            self.system_prompt.clone()
        } else {
            format!("local model reply: {last}")
        };
        Ok(ModelOutput::Text(text))
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

fn map_role(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "assistant",
    }
}

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ToolDefinition {
    r#type: String,
    function: ToolFunction,
}

#[derive(Debug, Serialize)]
struct ToolFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    function: CalledFunction,
}

#[derive(Debug, Deserialize)]
struct CalledFunction {
    name: String,
    arguments: String,
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
    use super::{ModelOutput, parse_structured_protocol};

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
}
