use serde_json::{Value, json};

use crate::types::{HandlerToLooperToolCallRequest, LooperToolDefinition, UserInputRequest};

pub const ASK_USER_QUESTION_TOOL_NAME: &str = "ask_user_question";

pub fn is_ask_user_question_tool(name: &str) -> bool {
    name == ASK_USER_QUESTION_TOOL_NAME
}

pub fn ask_user_question_tool_definition() -> LooperToolDefinition {
    LooperToolDefinition::default()
        .set_name(ASK_USER_QUESTION_TOOL_NAME)
        .set_description(
            "Ask the human user a question and wait for their response before continuing. \
             This tool MUST be called alone in its batch and should only be used when the model \
             genuinely needs human clarification or approval.",
        )
        .set_paramters(json!({
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to present to the user."
                },
                "options": {
                    "type": "array",
                    "description": "Optional list of suggested answer options.",
                    "items": { "type": "string" }
                },
                "context": {
                    "type": "string",
                    "description": "Optional extra context to help the user answer."
                }
            },
            "required": ["question"]
        }))
}

pub fn to_user_input_request(call: &HandlerToLooperToolCallRequest) -> UserInputRequest {
    UserInputRequest {
        id: call.id.clone(),
        question: call
            .args
            .get("question")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        options: call
            .args
            .get("options")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(ToString::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        context: call
            .args
            .get("context")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        raw_args: call.args.clone(),
    }
}

pub fn ask_user_question_batch_error() -> Value {
    json!({
        "error": "ask_user_question must be the only tool call in its batch"
    })
}
