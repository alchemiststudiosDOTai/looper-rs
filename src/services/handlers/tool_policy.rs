use serde_json::{Value, json};

use crate::{tools::LooperTools, types::ToolExecutionMode};

pub(crate) fn is_exclusive_tool(tools_runner: &dyn LooperTools, tool_name: &str) -> bool {
    matches!(
        tools_runner.get_tool_execution_mode(tool_name),
        ToolExecutionMode::Exclusive
    )
}

pub(crate) fn exclusive_tool_names<'a>(
    tools_runner: &dyn LooperTools,
    tool_names: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    tool_names
        .into_iter()
        .filter(|tool_name| is_exclusive_tool(tools_runner, tool_name))
        .map(ToString::to_string)
        .collect()
}

pub(crate) fn invalid_exclusive_tool_batch_result(exclusive_tool_names: &[String]) -> Value {
    json!({
        "error": format!(
            "Exclusive tool call(s) [{}] must be emitted alone in a single assistant turn.",
            exclusive_tool_names.join(", ")
        )
    })
}
