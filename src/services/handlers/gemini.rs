use std::sync::Arc;

use gemini_rust::{
    Content, FunctionResponse, Gemini, GenerationResponse, Message, Model, Part, Role, Tool,
};

use async_recursion::async_recursion;
use async_trait::async_trait;

use anyhow::Result;
use futures::TryStreamExt;

use tokio::{
    sync::{
        Mutex,
        mpsc::{Receiver, Sender},
    },
    task::JoinSet,
};

use crate::{
    mapping::tools::gemini::to_gemini_tool,
    services::{
        StreamingChatHandler,
        human_in_the_loop::{await_user_input, emit_tool_call_complete, emit_tool_call_request},
    },
    tools::LooperTools,
    tools::{ask_user_question_batch_error, is_ask_user_question_tool},
    types::{
        HandlerToLooperMessage, HandlerToLooperToolCallRequest, LooperToHandlerToolCallResult,
        LooperToolDefinition, MessageHistory,
    },
};

pub struct GeminiHandler {
    client: Gemini,
    system_message: String,
    messages: Vec<Message>,
    sender: Sender<HandlerToLooperMessage>,
    tool: Option<Tool>,
    user_response_receiver: Option<Arc<Mutex<Receiver<LooperToHandlerToolCallResult>>>>,
}

impl GeminiHandler {
    pub fn new(
        sender: Sender<HandlerToLooperMessage>,
        model: &str,
        system_message: &str,
        user_response_receiver: Option<Arc<Mutex<Receiver<LooperToHandlerToolCallResult>>>>,
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
            user_response_receiver,
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

        let mut tool_calls: Vec<HandlerToLooperToolCallRequest> = Vec::new();

        for (fc, thought_sig, tool_id) in &function_calls {
            tool_calls.push(HandlerToLooperToolCallRequest {
                id: tool_id.clone(),
                name: fc.name.clone(),
                args: fc.args.clone(),
            });

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

        if !tool_calls.is_empty() {
            if tool_calls
                .iter()
                .any(|call| is_ask_user_question_tool(&call.name))
            {
                let mut function_response_parts: Vec<Part> = Vec::new();

                if tool_calls.len() > 1 {
                    for tool_call in tool_calls {
                        emit_tool_call_request(&self.sender, &tool_call).await?;
                        emit_tool_call_complete(&self.sender, tool_call.id.clone()).await?;

                        function_response_parts.push(Part::FunctionResponse {
                            function_response: FunctionResponse {
                                name: tool_call.name,
                                response: Some(ask_user_question_batch_error()),
                            },
                        });
                    }
                } else {
                    let tool_call = tool_calls.into_iter().next().unwrap();
                    let response =
                        await_user_input(&self.sender, &self.user_response_receiver, &tool_call)
                            .await?;

                    emit_tool_call_complete(&self.sender, tool_call.id.clone()).await?;

                    function_response_parts.push(Part::FunctionResponse {
                        function_response: FunctionResponse {
                            name: tool_call.name,
                            response: Some(response),
                        },
                    });
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

            let mut tool_join_set = JoinSet::new();

            for tool_call in tool_calls {
                emit_tool_call_request(&self.sender, &tool_call).await?;

                let tr = tools_runner.clone();
                let tool_name = tool_call.name.clone();
                let tool_input = tool_call.args.clone();
                let tool_request = tool_call.clone();

                tool_join_set.spawn(async move {
                    let result = tr.run_tool(tool_name, tool_input).await;
                    (result, tool_request)
                });
            }

            let mut function_response_parts: Vec<Part> = Vec::new();

            while let Some(result) = tool_join_set.join_next().await {
                match result {
                    Ok((result, tool_use)) => {
                        emit_tool_call_complete(&self.sender, tool_use.id.clone()).await?;

                        function_response_parts.push(Part::FunctionResponse {
                            function_response: FunctionResponse {
                                name: tool_use.name.clone(),
                                response: Some(result),
                            },
                        });
                    }
                    Err(e) => {
                        eprintln!(
                            "Join Error occured when collecting tool call results | Error: {}",
                            e
                        );
                    }
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
