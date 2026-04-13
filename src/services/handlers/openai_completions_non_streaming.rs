use std::sync::Arc;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestToolMessage, ChatCompletionRequestUserMessageArgs,
        ChatCompletionTools, CreateChatCompletionRequestArgs, FinishReason, ReasoningEffort,
    },
};

use async_recursion::async_recursion;
use async_trait::async_trait;

use anyhow::Result;
use serde_json::Value;
use tokio::task::JoinSet;

use crate::{
    services::{ChatHandler, handlers::tool_policy::*},
    tools::LooperTools,
    types::{
        LooperToolDefinition, MessageHistory,
        turn::{ToolCallRecord, TurnResult, TurnStep},
    },
};

pub struct OpenAINonStreamingChatHandler {
    client: Client<OpenAIConfig>,
    model: String,
    messages: Vec<ChatCompletionRequestMessage>,
    tools: Vec<ChatCompletionTools>,
}

fn parse_tool_call_args(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn tool_call_name(tool_call: &ChatCompletionMessageToolCalls) -> &str {
    match tool_call {
        ChatCompletionMessageToolCalls::Function(func_call) => func_call.function.name.as_str(),
        ChatCompletionMessageToolCalls::Custom(custom_call) => {
            custom_call.custom_tool.name.as_str()
        }
    }
}

fn into_tool_call_parts(tool_call: ChatCompletionMessageToolCalls) -> (String, String, Value) {
    match tool_call {
        ChatCompletionMessageToolCalls::Function(func_call) => (
            func_call.id,
            func_call.function.name,
            parse_tool_call_args(&func_call.function.arguments),
        ),
        ChatCompletionMessageToolCalls::Custom(custom_call) => (
            custom_call.id,
            custom_call.custom_tool.name,
            parse_tool_call_args(&custom_call.custom_tool.input),
        ),
    }
}

impl OpenAINonStreamingChatHandler {
    pub fn new(model: &str, system_message: &str) -> Result<Self> {
        let client = Client::new();
        let system_message = ChatCompletionRequestSystemMessageArgs::default()
            .content(system_message)
            .build()?
            .into();

        Ok(OpenAINonStreamingChatHandler {
            client,
            model: model.to_string(),
            messages: vec![system_message],
            tools: Vec::new(),
        })
    }

    #[async_recursion]
    async fn inner_send_message(
        &mut self,
        tools_runner: Arc<dyn LooperTools>,
        steps: &mut Vec<TurnStep>,
    ) -> Result<()> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .max_completion_tokens(50000u32)
            .messages(self.messages.clone())
            .tools(self.tools.clone())
            .reasoning_effort(ReasoningEffort::Low)
            .build()?;

        let response = self.client.chat().create(request).await?;

        let choice = match response.choices.into_iter().next() {
            Some(c) => c,
            None => {
                steps.push(TurnStep {
                    thinking: Vec::new(),
                    text: None,
                    tool_calls: Vec::new(),
                });
                return Ok(());
            }
        };

        let message = choice.message;
        let text = message.content.clone();
        let has_tool_calls = matches!(choice.finish_reason, Some(FinishReason::ToolCalls));

        if has_tool_calls {
            let tool_calls_list = message.tool_calls.clone().unwrap_or_default();

            self.messages.push(
                ChatCompletionRequestAssistantMessage {
                    content: message.content.clone().map(|c| c.into()),
                    tool_calls: Some(tool_calls_list.clone()),
                    ..Default::default()
                }
                .into(),
            );

            let function_names = tool_calls_list
                .iter()
                .map(tool_call_name)
                .collect::<Vec<_>>();
            let exclusive_names = exclusive_tool_names(tools_runner.as_ref(), function_names);
            let mut tool_call_records = Vec::new();

            if !exclusive_names.is_empty() && tool_calls_list.len() > 1 {
                let error_result = invalid_exclusive_tool_batch_result(&exclusive_names);

                for tool_call in tool_calls_list {
                    let (tool_call_id, tool_name, args) = into_tool_call_parts(tool_call);

                    tool_call_records.push(ToolCallRecord {
                        id: tool_call_id.clone(),
                        name: tool_name,
                        args,
                        result: error_result.clone(),
                    });

                    self.messages.push(
                        ChatCompletionRequestToolMessage {
                            content: error_result.to_string().into(),
                            tool_call_id,
                        }
                        .into(),
                    );
                }
            } else if !exclusive_names.is_empty() {
                let tool_call = tool_calls_list
                    .into_iter()
                    .next()
                    .expect("tool call missing");
                let (tool_call_id, tool_name, args) = into_tool_call_parts(tool_call);
                let result = tools_runner.run_tool(tool_name.clone(), args.clone()).await;

                tool_call_records.push(ToolCallRecord {
                    id: tool_call_id.clone(),
                    name: tool_name,
                    args,
                    result: result.clone(),
                });

                self.messages.push(
                    ChatCompletionRequestToolMessage {
                        content: result.to_string().into(),
                        tool_call_id,
                    }
                    .into(),
                );
            } else {
                let mut tool_join_set = JoinSet::new();
                let mut ordered_results = Vec::new();
                ordered_results.resize_with(tool_calls_list.len(), || None);

                for (index, tool_call) in tool_calls_list.into_iter().enumerate() {
                    let (tool_call_id, tool_name, args) = into_tool_call_parts(tool_call);
                    let tr = tools_runner.clone();

                    tool_join_set.spawn(async move {
                        let result = tr.run_tool(tool_name.clone(), args.clone()).await;
                        (index, result, tool_call_id, tool_name, args)
                    });
                }

                while let Some(result) = tool_join_set.join_next().await {
                    match result {
                        Ok((index, result, tool_call_id, tool_name, args)) => {
                            ordered_results[index] = Some((result, tool_call_id, tool_name, args));
                        }
                        Err(e) => {
                            eprintln!(
                                "Join Error occured when collecting tool call results | Error: {}",
                                e
                            );
                        }
                    }
                }

                for (result, tool_call_id, tool_name, args) in ordered_results.into_iter().flatten()
                {
                    tool_call_records.push(ToolCallRecord {
                        id: tool_call_id.clone(),
                        name: tool_name,
                        args,
                        result: result.clone(),
                    });

                    self.messages.push(
                        ChatCompletionRequestToolMessage {
                            content: result.to_string().into(),
                            tool_call_id,
                        }
                        .into(),
                    );
                }
            }

            steps.push(TurnStep {
                thinking: Vec::new(),
                text,
                tool_calls: tool_call_records,
            });

            return self.inner_send_message(tools_runner, steps).await;
        }

        if let Some(ref content) = text {
            let assistant_msg = ChatCompletionRequestAssistantMessage {
                content: Some(content.clone().into()),
                ..Default::default()
            };
            self.messages.push(assistant_msg.into());
        }

        steps.push(TurnStep {
            thinking: Vec::new(),
            text,
            tool_calls: Vec::new(),
        });

        Ok(())
    }
}

#[async_trait]
impl ChatHandler for OpenAINonStreamingChatHandler {
    async fn send_message(
        &mut self,
        message_history: Option<MessageHistory>,
        message: &str,
        tools_runner: Arc<dyn LooperTools>,
    ) -> Result<TurnResult> {
        if let Some(MessageHistory::Messages(m)) = message_history {
            let messages: Vec<ChatCompletionRequestMessage> = serde_json::from_value(m)?;
            self.messages = messages;
        }

        let user_message = ChatCompletionRequestUserMessageArgs::default()
            .content(message)
            .build()?
            .into();

        self.messages.push(user_message);

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
        self.tools = tools
            .into_iter()
            .map(|t| ChatCompletionTools::Function(t.into()))
            .collect();
    }
}
