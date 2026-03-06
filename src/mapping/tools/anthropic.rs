use serde_json::{Map, Value, json};
use crate::types::LooperToolDefinition;

impl From<LooperToolDefinition> for Map<String, Value> {
    fn from(value: LooperToolDefinition) -> Self {
        let tool = json!({
            "name": value.name,
            "description": value.description,
            "input_schema": value.parameters
        });

        tool.as_object()
            .expect("Failed to build Anthropic tool Map from LooperToolDefinition")
            .to_owned()
    }
}
