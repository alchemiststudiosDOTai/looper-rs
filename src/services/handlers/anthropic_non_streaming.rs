use std::sync::Arc;

use async_anthropic::{
    Client,
    types::{
        CreateMessagesRequestBuilder, Message, MessageContent, MessageContentList, MessageRole,
        Thinking, Tool, ToolResultBuilder,
    },
};

use async_recursion::async_recursion;
use async_trait::async_trait;

use anyhow::Result;
use tokio::task::JoinSet;

use crate::{
    services::{ChatHandler, handlers::tool_policy::*},
    tools::LooperTools,
    types::{
        LooperToolDefinition, MessageHistory,
        turn::{ThinkingBlock, ToolCallRecord, TurnResult, TurnStep},
    },
};

pub struct AnthropicNonStreamingHandler {
    client: Client,
    model: String,
    system_message: String,
    messages: Vec<Message>,
    tools: Vec<Tool>,
}

impl AnthropicNonStreamingHandler {
    pub fn new(model: &str, system_message: &str) -> Result<Self> {
        let client = Client::default();

        Ok(AnthropicNonStreamingHandler {
            client,
            model: model.to_string(),
            system_message: system_message.to_string(),
            messages: vec![],
            tools: Vec::new(),
        })
    }

    #[async_recursion]
    async fn inner_send_message(
        &mut self,
        tools_runner: Arc<dyn LooperTools>,
        steps: &mut Vec<TurnStep>,
    ) -> Result<()> {
        let request = CreateMessagesRequestBuilder::default()
            .model(&self.model)
            .system(self.system_message.clone())
            .messages(self.messages.clone())
            .tools(self.tools.clone())
            .max_tokens(16384)
            .thinking(Thinking::Adaptive)
            .build()?;

        let response = self.client.messages().create(request).await?;

        let mut thinking = Vec::new();
        let mut text = None;
        let mut tool_uses = Vec::new();
        let mut assistant_content: Vec<MessageContent> = Vec::new();

        if let Some(content) = &response.content {
            for block in content {
                match block {
                    MessageContent::Thinking(t) => {
                        thinking.push(ThinkingBlock {
                            content: t.thinking.clone(),
                        });
                        assistant_content.push(block.clone());
                    }
                    MessageContent::Text(t) => {
                        text = Some(t.text.clone());
                        assistant_content.push(block.clone());
                    }
                    MessageContent::ToolUse(t) => {
                        tool_uses.push(t.clone());
                        assistant_content.push(block.clone());
                    }
                    _ => {
                        assistant_content.push(block.clone());
                    }
                }
            }
        }

        if !assistant_content.is_empty() {
            self.messages.push(Message {
                role: MessageRole::Assistant,
                content: MessageContentList(assistant_content),
            });
        }

        let mut tool_call_records = Vec::new();

        if !tool_uses.is_empty() {
            let exclusive_names = exclusive_tool_names(
                tools_runner.as_ref(),
                tool_uses.iter().map(|tool| tool.name.as_str()),
            );

            if !exclusive_names.is_empty() && tool_uses.len() > 1 {
                let error_result = invalid_exclusive_tool_batch_result(&exclusive_names);

                for tool_use in tool_uses {
                    tool_call_records.push(ToolCallRecord {
                        id: tool_use.id.clone(),
                        name: tool_use.name.clone(),
                        args: tool_use.input.clone(),
                        result: error_result.clone(),
                    });

                    self.messages.push(Message {
                        role: MessageRole::User,
                        content: MessageContentList(vec![MessageContent::ToolResult(
                            ToolResultBuilder::default()
                                .tool_use_id(&tool_use.id)
                                .content(error_result.to_string())
                                .build()?,
                        )]),
                    });
                }
            } else if !exclusive_names.is_empty() {
                let tool_use = tool_uses.into_iter().next().expect("tool use missing");
                let result = tools_runner
                    .run_tool(tool_use.name.clone(), tool_use.input.clone())
                    .await;

                tool_call_records.push(ToolCallRecord {
                    id: tool_use.id.clone(),
                    name: tool_use.name.clone(),
                    args: tool_use.input.clone(),
                    result: result.clone(),
                });

                self.messages.push(Message {
                    role: MessageRole::User,
                    content: MessageContentList(vec![MessageContent::ToolResult(
                        ToolResultBuilder::default()
                            .tool_use_id(&tool_use.id)
                            .content(result.to_string())
                            .build()?,
                    )]),
                });
            } else {
                let mut tool_join_set = JoinSet::new();
                let mut ordered_results = Vec::new();
                ordered_results.resize_with(tool_uses.len(), || None);

                for (index, tool_use) in tool_uses.into_iter().enumerate() {
                    let tr = tools_runner.clone();
                    tool_join_set.spawn(async move {
                        let result = tr
                            .run_tool(tool_use.name.clone(), tool_use.input.clone())
                            .await;

                        (index, result, tool_use)
                    });
                }

                while let Some(result) = tool_join_set.join_next().await {
                    match result {
                        Ok((index, result, tool_use)) => {
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
                    tool_call_records.push(ToolCallRecord {
                        id: tool_use.id.clone(),
                        name: tool_use.name.clone(),
                        args: tool_use.input.clone(),
                        result: result.clone(),
                    });

                    self.messages.push(Message {
                        role: MessageRole::User,
                        content: MessageContentList(vec![MessageContent::ToolResult(
                            ToolResultBuilder::default()
                                .tool_use_id(&tool_use.id)
                                .content(result.to_string())
                                .build()?,
                        )]),
                    });
                }
            }

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
impl ChatHandler for AnthropicNonStreamingHandler {
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

        self.messages.push(Message {
            role: MessageRole::User,
            content: MessageContentList(vec![MessageContent::from(message)]),
        });

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
        self.tools = tools.into_iter().map(|t| t.into()).collect();
    }
}
