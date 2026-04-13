use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{
    tools::{LooperTool, LooperTools},
    types::{LooperToolDefinition, ToolExecutionMode},
};

struct RegisteredTool {
    tool: Mutex<Arc<dyn LooperTool>>,
    execution_mode: ToolExecutionMode,
}

pub struct CompositeToolSet {
    base: Option<Box<dyn LooperTools>>,
    injected: HashMap<String, RegisteredTool>,
}

impl CompositeToolSet {
    pub fn new(base: Option<Box<dyn LooperTools>>) -> Self {
        CompositeToolSet {
            base,
            injected: HashMap::new(),
        }
    }
}

#[async_trait]
impl LooperTools for CompositeToolSet {
    async fn get_tools(&self) -> Vec<LooperToolDefinition> {
        let mut tools = if let Some(base) = &self.base {
            base.get_tools().await
        } else {
            Vec::new()
        };

        tools.retain(|tool| !self.injected.contains_key(&tool.name));

        for registered in self.injected.values() {
            let guard = registered.tool.lock().await;
            tools.push(guard.tool().clone());
        }

        tools
    }

    async fn add_tool(&mut self, tool: Arc<dyn LooperTool>) {
        let tool_name = tool.get_tool_name();
        let execution_mode = tool.execution_mode();

        self.injected.insert(
            tool_name,
            RegisteredTool {
                tool: Mutex::new(tool),
                execution_mode,
            },
        );
    }

    async fn run_tool(&self, name: String, args: Value) -> Value {
        if let Some(registered) = self.injected.get(&name) {
            let mut guard = registered.tool.lock().await;
            let tool = Arc::get_mut(&mut guard)
                .expect("tool has multiple references; injected tools must be uniquely owned");

            return tool.execute(&args).await;
        }

        if let Some(base) = &self.base {
            return base.run_tool(name, args).await;
        }

        json!({ "error": format!("Unknown function: {}", name) })
    }

    fn get_tool_execution_mode(&self, name: &str) -> ToolExecutionMode {
        if let Some(registered) = self.injected.get(name) {
            return registered.execution_mode;
        }

        self.base
            .as_deref()
            .map(|base| base.get_tool_execution_mode(name))
            .unwrap_or_default()
    }
}
