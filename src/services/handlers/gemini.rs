use std::sync::Arc;

use gemini_rust::{
    Content, FunctionResponse, Gemini, GenerationResponse, Message, Model, Part, Role, Tool,
};

use async_recursion::async_recursion;
use async_trait::async_trait;

use anyhow::Result;
use futures::TryStreamExt;
use tokio::{sync::mpsc::Sender, task::JoinSet};

use crate::{
    mapping::tools::gemini::to_gemini_tool,
    services::{StreamingChatHandler, handlers::tool_policy::*},
    tools::LooperTools,
    types::{
        HandlerToLooperMessage, HandlerToLooperToolCallRequest, LooperToolDefinition,
        MessageHistory,
    },
};

pub struct GeminiHandler {
    client: Gemini,
    system_message: String,
    messages: Vec<Message>,
    sender: Sender<HandlerToLooperMessage>,
    tool: Option<Tool>,
}

impl GeminiHandler {
    pub fn new(
        sender: Sender<HandlerToLooperMessage>,
        model: &str,
        system_message: &str,
    ) -> Result<Self> {
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

        Ok(GeminiHandler {
            client,
            system_message: system_message.to_string(),
            messages: vec![],
            sender,
            tool: None,
        })
    }

    #[async_recursion]
    async fn inner_send_message(&mut self, tools_runner: Arc<dyn LooperTools>) -> Result<()> {
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

        let mut stream = builder.execute_stream().await?;

        let mut all_text = String::new();
        let mut thinking_text = String::new();
        let mut function_calls: Vec<(gemini_rust::FunctionCall, Option<String>, String)> =
            Vec::new();
        let mut had_thinking = false;

        while let Some(chunk) = stream.try_next().await? {
            self.process_stream_chunk(
                &chunk,
                &mut all_text,
                &mut thinking_text,
                &mut function_calls,
                &mut had_thinking,
            )
            .await?;
        }

        if had_thinking {
            self.sender
                .send(HandlerToLooperMessage::ThinkingComplete)
                .await?;
        }

        let mut assistant_parts: Vec<Part> = Vec::new();

        if !thinking_text.is_empty() {
            assistant_parts.push(Part::Text {
                text: thinking_text,
                thought: Some(true),
                thought_signature: None,
            });
        }

        if !all_text.is_empty() {
            assistant_parts.push(Part::Text {
                text: all_text,
                thought: None,
                thought_signature: None,
            });
        }

        for (fc, thought_sig, _tool_id) in &function_calls {
            assistant_parts.push(Part::FunctionCall {
                function_call: fc.clone(),
                thought_signature: thought_sig.clone(),
            });
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

        if !function_calls.is_empty() {
            let exclusive_names = exclusive_tool_names(
                tools_runner.as_ref(),
                function_calls.iter().map(|(fc, _, _)| fc.name.as_str()),
            );
            let mut function_response_parts: Vec<Part> = Vec::new();

            if !exclusive_names.is_empty() && function_calls.len() > 1 {
                let error_result = invalid_exclusive_tool_batch_result(&exclusive_names);

                for (fc, _thought_sig, tool_id) in function_calls {
                    let tcr = HandlerToLooperToolCallRequest {
                        id: tool_id.clone(),
                        name: fc.name.clone(),
                        args: fc.args.clone(),
                    };

                    self.sender
                        .send(HandlerToLooperMessage::ToolCallRequest(tcr))
                        .await?;
                    self.sender
                        .send(HandlerToLooperMessage::ToolCallComplete(tool_id))
                        .await?;

                    function_response_parts.push(Part::FunctionResponse {
                        function_response: FunctionResponse {
                            name: fc.name.clone(),
                            response: Some(error_result.clone()),
                        },
                    });
                }
            } else if !exclusive_names.is_empty() {
                let (fc, _thought_sig, tool_id) = function_calls
                    .into_iter()
                    .next()
                    .expect("function call missing");
                let tcr = HandlerToLooperToolCallRequest {
                    id: tool_id.clone(),
                    name: fc.name.clone(),
                    args: fc.args.clone(),
                };

                self.sender
                    .send(HandlerToLooperMessage::ToolCallRequest(tcr))
                    .await?;

                let result = tools_runner
                    .run_tool(fc.name.clone(), fc.args.clone())
                    .await;

                self.sender
                    .send(HandlerToLooperMessage::ToolCallComplete(tool_id))
                    .await?;

                function_response_parts.push(Part::FunctionResponse {
                    function_response: FunctionResponse {
                        name: fc.name.clone(),
                        response: Some(result),
                    },
                });
            } else {
                let mut tool_join_set = JoinSet::new();
                let mut ordered_results = Vec::new();
                ordered_results.resize_with(function_calls.len(), || None);

                for (index, (fc, _thought_sig, tool_id)) in function_calls.into_iter().enumerate() {
                    let tcr = HandlerToLooperToolCallRequest {
                        id: tool_id.clone(),
                        name: fc.name.clone(),
                        args: fc.args.clone(),
                    };

                    self.sender
                        .send(HandlerToLooperMessage::ToolCallRequest(tcr.clone()))
                        .await?;

                    let tr = tools_runner.clone();
                    tool_join_set.spawn(async move {
                        let result = tr.run_tool(fc.name.clone(), fc.args.clone()).await;
                        (index, result, tcr)
                    });
                }

                while let Some(result) = tool_join_set.join_next().await {
                    match result {
                        Ok((index, result, tool_use)) => {
                            self.sender
                                .send(HandlerToLooperMessage::ToolCallComplete(
                                    tool_use.id.clone(),
                                ))
                                .await?;
                            ordered_results[index] = Some((result, tool_use));
                        }
                        Err(e) => {
                            eprintln!(
                                "Join Error occured when collecting tool call results | Error: {}",
                                e
                            );
                        }
                    }
                }

                for (result, tool_use) in ordered_results.into_iter().flatten() {
                    function_response_parts.push(Part::FunctionResponse {
                        function_response: FunctionResponse {
                            name: tool_use.name.clone(),
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

            return self.inner_send_message(tools_runner).await;
        }

        Ok(())
    }

    async fn process_stream_chunk(
        &self,
        chunk: &GenerationResponse,
        all_text: &mut String,
        thinking_text: &mut String,
        function_calls: &mut Vec<(gemini_rust::FunctionCall, Option<String>, String)>,
        had_thinking: &mut bool,
    ) -> Result<()> {
        for candidate in &chunk.candidates {
            if let Some(parts) = &candidate.content.parts {
                for part in parts {
                    match part {
                        Part::Text {
                            text,
                            thought,
                            thought_signature: _,
                        } => {
                            if *thought == Some(true) {
                                *had_thinking = true;
                                thinking_text.push_str(text);
                                if !text.is_empty() {
                                    self.sender
                                        .send(HandlerToLooperMessage::Thinking(text.clone()))
                                        .await?;
                                }
                            } else if !text.is_empty() {
                                all_text.push_str(text);
                                self.sender
                                    .send(HandlerToLooperMessage::Assistant(text.clone()))
                                    .await?;
                            }
                        }
                        Part::FunctionCall {
                            function_call,
                            thought_signature,
                        } => {
                            let tool_id = uuid::Uuid::new_v4().to_string();
                            self.sender
                                .send(HandlerToLooperMessage::ToolCallPending(tool_id.clone()))
                                .await?;
                            function_calls.push((
                                function_call.clone(),
                                thought_signature.clone(),
                                tool_id,
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl StreamingChatHandler for GeminiHandler {
    async fn send_message(
        &mut self,
        message_history: Option<MessageHistory>,
        message: &str,
        tools_runner: Arc<dyn LooperTools>,
    ) -> Result<MessageHistory> {
        if let Some(MessageHistory::Messages(m)) = message_history {
            let messages: Vec<Message> = serde_json::from_value(m)?;
            self.messages = messages;
        }

        self.messages.push(Message::user(message));

        self.inner_send_message(tools_runner).await?;

        self.sender
            .send(HandlerToLooperMessage::TurnComplete)
            .await?;

        let messages = serde_json::to_value(&self.messages)?;

        Ok(MessageHistory::Messages(messages))
    }

    fn set_tools(&mut self, tools: Vec<LooperToolDefinition>) {
        if tools.is_empty() {
            self.tool = None;
        } else {
            self.tool = Some(to_gemini_tool(tools));
        }
    }
}
