use serde_json::Value;
use tokio::sync::oneshot::Sender;

type Name = String;
type Message = String;
type ToolIndex = usize;

#[derive(Debug)]
pub enum HandlerToLooperMessage {
    Assistant(Message),
    Thinking(Message),
    ThinkingComplete,
    ToolCallPending(ToolIndex),
    ToolCallRequest(HandlerToLooperToolCallRequest),
    TurnComplete
}

#[derive(Debug)]
pub struct HandlerToLooperToolCallRequest {
    pub id: String,
    pub name: String,
    pub args: Value,
    pub tool_result_channel: Sender<LooperToHandlerToolCallResult>
}

#[derive(Debug)]
pub struct LooperToHandlerToolCallResult {
    pub id: String,
    pub value: Value
}

#[derive(Debug)]
pub enum LooperToInterfaceMessage {
    Assistant(Message),
    Thinking(Message),
    ThinkingComplete,
    ToolCallPending(ToolIndex),
    ToolCall(Name),
    TurnComplete
}
