use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};

use crate::{tools::LooperTool, types::LooperToolDefinition};

#[async_trait]
pub trait AskUserHandler: Send + Sync {
    async fn ask_user(&self, question: String, context: Option<String>) -> Result<String>;
}

pub struct AskUserTool {
    handler: Arc<dyn AskUserHandler>,
}

impl AskUserTool {
    pub fn new(handler: Arc<dyn AskUserHandler>) -> Self {
        AskUserTool { handler }
    }
}

#[async_trait]
impl LooperTool for AskUserTool {
    fn get_tool_name(&self) -> String { "ask_user".to_string() }

    fn tool(&self) -> LooperToolDefinition {
        LooperToolDefinition::default()
            .set_name("ask_user")
            .set_description(
                "Ask the human user a question when required information is missing or must come from the user directly. Use this only for information the available tools cannot determine themselves.",
            )
            .set_paramters(json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "The exact question to ask the user." },
                    "context": { "type": "string", "description": "Optional brief context explaining why the information is needed." }
                },
                "required": ["question"]
            }))
    }

    async fn execute(&mut self, args: &Value) -> Value {
        let Some(question) = args["question"].as_str() else {
            return json!({ "error": "Missing 'question' argument" });
        };

        let context = args["context"].as_str().map(str::to_string);

        match self.handler.ask_user(question.to_string(), context.clone()).await {
            Ok(answer) => json!({
                "question": question,
                "context": context,
                "answer": answer,
            }),
            Err(error) => json!({
                "question": question,
                "context": context,
                "error": error.to_string(),
            }),
        }
    }
}
