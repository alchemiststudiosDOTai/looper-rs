use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::{
    tools::LooperTool,
    types::{AskUserRequest, LooperToolDefinition, ToolExecutionMode},
};

pub struct AskUserTool {
    sender: crate::types::AskUserSender,
}

impl AskUserTool {
    pub const NAME: &'static str = "ask_user";

    pub fn new(sender: crate::types::AskUserSender) -> Self {
        AskUserTool { sender }
    }
}

#[async_trait]
impl LooperTool for AskUserTool {
    fn get_tool_name(&self) -> String {
        Self::NAME.to_string()
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Exclusive
    }

    fn tool(&self) -> LooperToolDefinition {
        LooperToolDefinition::default()
            .set_name(Self::NAME)
            .set_description(
                "Ask the human user a blocking question when you need input to continue. IMPORTANT: call this tool alone, never in the same assistant turn as any other tool.",
            )
            .set_paramters(json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The exact question to show the user."
                    },
                    "options": {
                        "type": "array",
                        "description": "Optional suggested answer choices to show the user.",
                        "items": { "type": "string" }
                    }
                },
                "required": ["question"]
            }))
    }

    async fn execute(&mut self, args: &Value) -> Value {
        let Some(question) = args.get("question").and_then(Value::as_str) else {
            return json!({ "error": "Missing 'question' argument" });
        };

        let options = args
            .get("options")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(ToString::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let (response_tx, response_rx) = oneshot::channel();
        let request = AskUserRequest {
            id: Uuid::new_v4().to_string(),
            question: question.to_string(),
            options,
            response_tx,
        };

        if self.sender.send(request).await.is_err() {
            return json!({ "error": "ask_user channel is closed" });
        }

        match response_rx.await {
            Ok(response) => json!({ "answer": response.answer }),
            Err(_) => json!({ "error": "ask_user response channel was dropped" }),
        }
    }
}
