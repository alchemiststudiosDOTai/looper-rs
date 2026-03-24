use serde::{Deserialize, Serialize};
use serde_json::Value;

type Name = String;
type Message = String;
type ToolId = String;

#[derive(Debug)]
pub enum HandlerToLooperMessage {
    Assistant(Message),
    Thinking(Message),
    ThinkingComplete,
    ToolCallPending(ToolId),
    ToolCallRequest(HandlerToLooperToolCallRequest),
    UserInputRequest(UserInputRequest),
    ToolCallComplete(ToolId),
    TurnComplete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerToLooperToolCallRequest {
    pub id: String,
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LooperToHandlerToolCallResult {
    pub id: String,
    pub value: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInputRequest {
    pub id: String,
    pub question: String,
    pub options: Vec<String>,
    pub context: Option<String>,
    pub raw_args: Value,
}

#[derive(Debug)]
pub enum LooperToInterfaceMessage {
    Assistant(Message),
    Thinking(Message),
    ThinkingComplete,
    ToolCallPending(ToolId),
    ToolCall(Name),
    UserInputRequest(UserInputRequest),
    ToolCallComplete(ToolId),
    TurnComplete,
}
