pub mod ask_user_question;
pub use ask_user_question::*;

pub mod sub_agent;
pub use sub_agent::*;

pub mod empty;
pub use empty::*;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::types::LooperToolDefinition;

#[async_trait]
pub trait LooperTool: Send + Sync {
    async fn execute(&mut self, args: &Value) -> Value;
    fn tool(&self) -> LooperToolDefinition;
    fn get_tool_name(&self) -> String;
}

#[async_trait]
pub trait LooperTools: Send + Sync {
    async fn get_tools(&self) -> Vec<LooperToolDefinition>;
    async fn add_tool(&mut self, tool: Arc<dyn LooperTool>);
    async fn run_tool(&self, name: String, args: Value) -> Value;
}
