use gemini_rust::{FunctionDeclaration, Tool};
use serde_json::json;

use crate::types::LooperToolDefinition;

pub fn to_gemini_tool(tools: Vec<LooperToolDefinition>) -> Tool {
    let declarations: Vec<FunctionDeclaration> = tools
        .into_iter()
        .map(|t| {
            // FunctionDeclaration's parameters fields are pub(crate), so we
            // construct via serde deserialization to set parametersJsonSchema
            // from our existing JSON Schema value.
            let fd_value = json!({
                "name": t.name,
                "description": t.description,
                "parametersJsonSchema": t.parameters,
            });
            serde_json::from_value(fd_value)
                .expect("Failed to construct FunctionDeclaration from LooperToolDefinition")
        })
        .collect();

    Tool::Function {
        function_declarations: declarations,
    }
}
