use std::sync::Arc;

use gemini_rust::{Content, FunctionResponse, Gemini, Message, Model, Part, Role, Tool};

use async_recursion::async_recursion;
use async_trait::async_trait;

use anyhow::Result;
use tokio::task::JoinSet;

use crate::{
    mapping::tools::gemini::to_gemini_tool,
    services::{ChatHandler, handlers::tool_policy::*},
    tools::LooperTools,
    types::{
        LooperToolDefinition, MessageHistory,
        turn::{ThinkingBlock, ToolCallRecord, TurnResult, TurnStep},
    },
};

pub struct GeminiNonStreamingHandler {
    client: Gemini,
    system_message: String,
    messages: Vec<Message>,
    tool: Option<Tool>,
}

impl GeminiNonStreamingHandler {
    pub fn new(model: &str, system_message: &str) -> Result<Self> {
        let api_key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| {
                anyhow::anyhow!("GEMINI_API_KEY or GOOGLE_API_KEY environment variable must be set")
            })?;

        let model_id = if model.starts_with("models/") {
            Model::Custom(model.to_string())
        } else {
            Model::Custom(format!("models/{}", model))
        };
        let client = Gemini::with_model(&api_key, model_id)?;

        Ok(GeminiNonStreamingHandler {
            client,
            system_message: system_message.to_string(),
            messages: vec![],
            tool: None,
        })
    }

    #[async_recursion]
    async fn inner_send_message(
        &mut self,
        tools_runner: Arc<dyn LooperTools>,
        steps: &mut Vec<TurnStep>,
    ) -> Result<()> {
        let mut builder = self
            .client
            .generate_content()
            .with_system_prompt(&self.system_message)
            .with_messages(self.messages.clone())
            .with_thinking_budget(-1)
            .with_thoughts_included(true);

        if let Some(tool) = &self.tool {
            builder = builder.with_tool(tool.clone());
        }

        let response = builder.execute().await?;

        let mut thinking = Vec::new();
        let mut text = None;
        let mut func_calls: Vec<(gemini_rust::FunctionCall, Option<String>)> = Vec::new();
        let mut assistant_parts: Vec<Part> = Vec::new();

        for candidate in &response.candidates {
            if let Some(parts) = &candidate.content.parts {
                for part in parts {
                    match part {
                        Part::Text {
                            text: t,
                            thought,
                            thought_signature: _,
                        } => {
                            if *thought == Some(true) {
                                thinking.push(ThinkingBlock { content: t.clone() });
                            } else {
                                text = Some(t.clone());
                            }
                            assistant_parts.push(part.clone());
                        }
                        Part::FunctionCall {
                            function_call,
                            thought_signature,
                        } => {
                            func_calls.push((function_call.clone(), thought_signature.clone()));
                            assistant_parts.push(part.clone());
                        }
                        _ => {
                            assistant_parts.push(part.clone());
                        }
                    }
                }
            }
        }

        if !assistant_parts.is_empty() {
            self.messages.push(Message {
                content: Content {
                    parts: Some(assistant_parts),
                    role: Some(Role::Model),
                },
                role: Role::Model,
            });
        }

        let mut tool_call_records = Vec::new();

        if !func_calls.is_empty() {
            let exclusive_names = exclusive_tool_names(
                tools_runner.as_ref(),
                func_calls.iter().map(|(fc, _)| fc.name.as_str()),
            );
            let mut function_response_parts: Vec<Part> = Vec::new();

            if !exclusive_names.is_empty() && func_calls.len() > 1 {
                let error_result = invalid_exclusive_tool_batch_result(&exclusive_names);

                for (fc, _thought_sig) in func_calls {
                    let tool_id = uuid::Uuid::new_v4().to_string();
                    tool_call_records.push(ToolCallRecord {
                        id: tool_id,
                        name: fc.name.clone(),
                        args: fc.args.clone(),
                        result: error_result.clone(),
                    });

                    function_response_parts.push(Part::FunctionResponse {
                        function_response: FunctionResponse {
                            name: fc.name.clone(),
                            response: Some(error_result.clone()),
                        },
                    });
                }
            } else if !exclusive_names.is_empty() {
                let (fc, _thought_sig) = func_calls
                    .into_iter()
                    .next()
                    .expect("function call missing");
                let tool_id = uuid::Uuid::new_v4().to_string();
                let result = tools_runner
                    .run_tool(fc.name.clone(), fc.args.clone())
                    .await;

                tool_call_records.push(ToolCallRecord {
                    id: tool_id,
                    name: fc.name.clone(),
                    args: fc.args.clone(),
                    result: result.clone(),
                });

                function_response_parts.push(Part::FunctionResponse {
                    function_response: FunctionResponse {
                        name: fc.name.clone(),
                        response: Some(result),
                    },
                });
            } else {
                let mut tool_join_set = JoinSet::new();
                let mut ordered_results = Vec::new();
                ordered_results.resize_with(func_calls.len(), || None);

                for (index, (fc, _thought_sig)) in func_calls.into_iter().enumerate() {
                    let tr = tools_runner.clone();
                    let tool_id = uuid::Uuid::new_v4().to_string();
                    tool_join_set.spawn(async move {
                        let result = tr.run_tool(fc.name.clone(), fc.args.clone()).await;
                        (index, result, fc, tool_id)
                    });
                }

                while let Some(result) = tool_join_set.join_next().await {
                    match result {
                        Ok((index, result, fc, tool_id)) => {
                            ordered_results[index] = Some((result, fc, tool_id));
                        }
                        Err(e) => {
                            eprintln!(
                                "Join Error occured when collecting tool call results | Error: {}",
                                e
                            );
                        }
                    }
                }

                for (result, fc, tool_id) in ordered_results.into_iter().flatten() {
                    tool_call_records.push(ToolCallRecord {
                        id: tool_id,
                        name: fc.name.clone(),
                        args: fc.args.clone(),
                        result: result.clone(),
                    });

                    function_response_parts.push(Part::FunctionResponse {
                        function_response: FunctionResponse {
                            name: fc.name.clone(),
                            response: Some(result),
                        },
                    });
                }
            }

            self.messages.push(Message {
                content: Content {
                    parts: Some(function_response_parts),
                    role: Some(Role::User),
                },
                role: Role::User,
            });

            steps.push(TurnStep {
                thinking,
                text,
                tool_calls: tool_call_records,
            });

            return self.inner_send_message(tools_runner, steps).await;
        }

        steps.push(TurnStep {
            thinking,
            text,
            tool_calls: tool_call_records,
        });

        Ok(())
    }
}

#[async_trait]
impl ChatHandler for GeminiNonStreamingHandler {
    async fn send_message(
        &mut self,
        message_history: Option<MessageHistory>,
        message: &str,
        tools_runner: Arc<dyn LooperTools>,
    ) -> Result<TurnResult> {
        if let Some(MessageHistory::Messages(m)) = message_history {
            let messages: Vec<Message> = serde_json::from_value(m)?;
            self.messages = messages;
        }

        self.messages.push(Message::user(message));

        let mut steps = Vec::new();
        self.inner_send_message(tools_runner, &mut steps).await?;

        let final_text = steps.iter().rev().find_map(|s| s.text.clone());

        let message_history = MessageHistory::Messages(serde_json::to_value(&self.messages)?);

        Ok(TurnResult {
            steps,
            final_text,
            message_history,
        })
    }

    fn set_tools(&mut self, tools: Vec<LooperToolDefinition>) {
        if tools.is_empty() {
            self.tool = None;
        } else {
            self.tool = Some(to_gemini_tool(tools));
        }
    }
}
