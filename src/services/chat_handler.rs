use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use crate::types::LooperToolDefinition;

#[async_trait]
pub trait ChatHandler: Send + Sync {
    async fn send_message(
        &mut self, 
        message_history: Option<Value>, // serde_json::from_value requires owned value
        message: &str
    ) -> Result<Value>;

    fn set_tools(&mut self, tools: Vec<LooperToolDefinition>);
}
