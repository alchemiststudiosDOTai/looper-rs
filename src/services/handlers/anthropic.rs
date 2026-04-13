use std::{collections::HashMap, sync::Arc};

use async_anthropic::{
    Client,
    types::{
        ContentBlockDelta, CreateMessagesRequestBuilder, Message, MessageContent,
        MessageContentList, MessageRole, MessagesStreamEvent, Thinking, Tool, ToolResultBuilder,
    },
};

use async_recursion::async_recursion;
use async_trait::async_trait;

use anyhow::Result;
use futures::StreamExt;
use tokio::{sync::mpsc::Sender, task::JoinSet};

use crate::{
    services::{StreamingChatHandler, handlers::tool_policy::*},
    tools::LooperTools,
    types::{
        HandlerToLooperMessage, HandlerToLooperToolCallRequest, LooperToolDefinition,
        MessageHistory,
    },
};

pub struct AnthropicHandler {
    client: Client,
    model: String,
    system_message: String,
    messages: Vec<Message>,
    sender: Sender<HandlerToLooperMessage>,
    tools: Vec<Tool>,
}

impl AnthropicHandler {
    pub fn new(
        sender: Sender<HandlerToLooperMessage>,
        model: &str,
        system_message: &str,
    ) -> Result<Self> {
        let client = Client::default();

        Ok(AnthropicHandler {
            client,
            model: model.to_string(),
            system_message: system_message.to_string(),
            messages: vec![],
            sender,
            tools: Vec::new(),
        })
    }

    #[async_recursion]
    async fn inner_send_message(&mut self, tools_runner: Arc<dyn LooperTools>) -> Result<String> {
        let request = CreateMessagesRequestBuilder::default()
            .model(&self.model)
            .system(self.system_message.clone())
            .messages(self.messages.clone())
            .tools(self.tools.clone())
            .max_tokens(16384)
            .thinking(Thinking::Adaptive)
            .build()?;

        let mut stream = self.client.messages().create_stream(request).await;
        let mut content_blocks = HashMap::new();
        let mut tool_input_bufs: HashMap<usize, String> = HashMap::new();
        let mut signatures: HashMap<usize, String> = HashMap::new();

        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => match response {
                    MessagesStreamEvent::ContentBlockStart {
                        index,
                        content_block,
                    } => {
                        content_blocks.insert(index, content_block);
                    }
                    MessagesStreamEvent::ContentBlockDelta { index, delta } => {
                        if let Some(cb) = content_blocks.get_mut(&index) {
                            match delta {
                                ContentBlockDelta::TextDelta { text } => {
                                    if let MessageContent::Text(t) = cb {
                                        t.text += &text;

                                        self.sender
                                            .send(HandlerToLooperMessage::Assistant(text))
                                            .await?;
                                    }
                                }
                                ContentBlockDelta::ThinkingDelta { thinking } => {
                                    if let MessageContent::Thinking(t) = cb {
                                        t.thinking += &thinking;

                                        self.sender
                                            .send(HandlerToLooperMessage::Thinking(thinking))
                                            .await?;
                                    }
                                }
                                ContentBlockDelta::SignatureDelta { signature } => {
                                    if let MessageContent::Thinking(_) = cb {
                                        signatures.entry(index).or_default().push_str(&signature);
                                    }
                                }
                                ContentBlockDelta::InputJsonDelta { partial_json } => {
                                    if let MessageContent::ToolUse(t) = cb {
                                        tool_input_bufs
                                            .entry(index)
                                            .or_default()
                                            .push_str(&partial_json);

                                        self.sender
                                            .send(HandlerToLooperMessage::ToolCallPending(
                                                t.id.clone(),
                                            ))
                                            .await?;
                                    }
                                }
                            }
                        }
                    }
                    MessagesStreamEvent::ContentBlockStop { index } => {
                        if let Some(MessageContent::Thinking(_)) = content_blocks.get(&index) {
                            self.sender
                                .send(HandlerToLooperMessage::ThinkingComplete)
                                .await?;
                        }

                        if let Some(raw_input) = tool_input_bufs.remove(&index)
                            && let Some(MessageContent::ToolUse(t)) = content_blocks.get_mut(&index)
                            && !raw_input.is_empty()
                        {
                            t.input = serde_json::from_str(&raw_input)?;
                        }
                    }
                    _ => {}
                },
                Err(err) => {
                    println!("error: {err:?}");
                }
            }
        }

        let mut sorted_indices: Vec<usize> = content_blocks.keys().copied().collect();
        sorted_indices.sort();

        let mut assistant_content: Vec<MessageContent> = Vec::new();
        let mut tool_requests = Vec::new();

        for index in &sorted_indices {
            if let Some(mut block) = content_blocks.remove(index) {
                if let MessageContent::Thinking(ref mut t) = block
                    && let Some(sig) = signatures.remove(index)
                {
                    t.signature = Some(sig);
                }

                if let MessageContent::ToolUse(ref t) = block {
                    let tcr = HandlerToLooperToolCallRequest {
                        id: t.id.clone(),
                        name: t.name.clone(),
                        args: t.input.clone(),
                    };

                    self.sender
                        .send(HandlerToLooperMessage::ToolCallRequest(tcr.clone()))
                        .await?;
                    tool_requests.push(tcr);
                }

                assistant_content.push(block);
            }
        }

        if !assistant_content.is_empty() {
            self.messages.push(Message {
                role: MessageRole::Assistant,
                content: MessageContentList(assistant_content),
            });
        }

        if !tool_requests.is_empty() {
            let exclusive_names = exclusive_tool_names(
                tools_runner.as_ref(),
                tool_requests.iter().map(|request| request.name.as_str()),
            );

            if !exclusive_names.is_empty() && tool_requests.len() > 1 {
                let error_result = invalid_exclusive_tool_batch_result(&exclusive_names);

                for request in tool_requests {
                    self.sender
                        .send(HandlerToLooperMessage::ToolCallComplete(request.id.clone()))
                        .await?;

                    self.messages.push(Message {
                        role: MessageRole::User,
                        content: MessageContentList(vec![MessageContent::ToolResult(
                            ToolResultBuilder::default()
                                .tool_use_id(&request.id)
                                .content(error_result.to_string())
                                .build()?,
                        )]),
                    });
                }

                return self.inner_send_message(tools_runner).await;
            }

            if !exclusive_names.is_empty() {
                let request = tool_requests
                    .into_iter()
                    .next()
                    .expect("tool request missing");
                let result = tools_runner
                    .run_tool(request.name.clone(), request.args.clone())
                    .await;

                self.sender
                    .send(HandlerToLooperMessage::ToolCallComplete(request.id.clone()))
                    .await?;

                self.messages.push(Message {
                    role: MessageRole::User,
                    content: MessageContentList(vec![MessageContent::ToolResult(
                        ToolResultBuilder::default()
                            .tool_use_id(&request.id)
                            .content(result.to_string())
                            .build()?,
                    )]),
                });

                return self.inner_send_message(tools_runner).await;
            }

            let mut tool_join_set = JoinSet::new();
            let mut ordered_results = Vec::new();
            ordered_results.resize_with(tool_requests.len(), || None);

            for (index, request) in tool_requests.into_iter().enumerate() {
                let tr = tools_runner.clone();
                tool_join_set.spawn(async move {
                    let result = tr
                        .run_tool(request.name.clone(), request.args.clone())
                        .await;
                    (index, result, request)
                });
            }

            while let Some(result) = tool_join_set.join_next().await {
                match result {
                    Ok((index, result, request)) => {
                        self.sender
                            .send(HandlerToLooperMessage::ToolCallComplete(request.id.clone()))
                            .await?;
                        ordered_results[index] = Some((result, request));
                    }
                    Err(e) => {
                        eprintln!(
                            "Join Error occured when collecting tool call results | Error: {}",
                            e
                        );
                    }
                }
            }

            for (result, request) in ordered_results.into_iter().flatten() {
                self.messages.push(Message {
                    role: MessageRole::User,
                    content: MessageContentList(vec![MessageContent::ToolResult(
                        ToolResultBuilder::default()
                            .tool_use_id(&request.id)
                            .content(result.to_string())
                            .build()?,
                    )]),
                });
            }

            return self.inner_send_message(tools_runner).await;
        }

        Ok(String::new())
    }
}

#[async_trait]
impl StreamingChatHandler for AnthropicHandler {
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

        self.messages.push(Message {
            role: MessageRole::User,
            content: MessageContentList(vec![MessageContent::from(message)]),
        });

        self.inner_send_message(tools_runner).await?;

        self.sender
            .send(HandlerToLooperMessage::TurnComplete)
            .await?;

        let messages = serde_json::to_value(&self.messages)?;

        Ok(MessageHistory::Messages(messages))
    }

    fn set_tools(&mut self, tools: Vec<LooperToolDefinition>) {
        self.tools = tools.into_iter().map(|t| t.into()).collect();
    }
}
