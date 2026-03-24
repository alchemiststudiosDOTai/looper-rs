use std::sync::Arc;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        chat::ReasoningEffort,
        responses::{
            CreateResponseArgs, FunctionCallOutput, FunctionCallOutputItemParam, FunctionToolCall,
            InputItem, InputParam, Item, OutputItem, Reasoning, ReasoningSummary,
            ResponseStreamEvent, Tool,
        },
    },
};

use async_recursion::async_recursion;
use async_trait::async_trait;

use anyhow::Result;
use futures::StreamExt;
use serde_json::Value;
use tokio::{
    sync::{Mutex, mpsc::Receiver},
    task::JoinSet,
};

use crate::{
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

pub struct OpenAIResponsesHandler {
    client: Client<OpenAIConfig>,
    model: String,
    previous_response_id: Option<String>,
    sender: tokio::sync::mpsc::Sender<HandlerToLooperMessage>,
    tools: Vec<Tool>,
    instructions: String,
    user_response_receiver: Option<Arc<Mutex<Receiver<LooperToHandlerToolCallResult>>>>,
}

impl OpenAIResponsesHandler {
    pub fn new(
        sender: tokio::sync::mpsc::Sender<HandlerToLooperMessage>,
        model: &str,
        system_message: &str,
        user_response_receiver: Option<Arc<Mutex<Receiver<LooperToHandlerToolCallResult>>>>,
    ) -> Result<Self> {
        let client = Client::new();

        Ok(OpenAIResponsesHandler {
            client,
            model: model.to_string(),
            previous_response_id: None,
            sender,
            tools: Vec::new(),
            instructions: system_message.to_string(),
            user_response_receiver,
        })
    }

    #[async_recursion]
    async fn inner_send_message(
        &mut self,
        input: Option<InputParam>,
        tools_runner: Arc<dyn LooperTools>,
    ) -> Result<String> {
        let mut builder = CreateResponseArgs::default();
        builder
            .model(&self.model)
            .tools(self.tools.clone())
            .reasoning(Reasoning {
                effort: Some(ReasoningEffort::High),
                summary: Some(ReasoningSummary::Concise),
            })
            .instructions(self.instructions.clone());

        if let Some(i) = input {
            builder.input(i);
        }

        if let Some(ref prev_id) = self.previous_response_id {
            builder.previous_response_id(prev_id);
        }

        let request = builder.build()?;
        let mut stream = self.client.responses().create_stream(request).await?;

        let mut assistant_res_buf = Vec::new();
        let mut function_calls: Vec<FunctionToolCall> = Vec::new();
        let mut response_id: Option<String> = None;

        while let Some(event) = stream.next().await {
            match event {
                Ok(ResponseStreamEvent::ResponseOutputTextDelta(delta)) => {
                    let text = delta.delta.clone();
                    assistant_res_buf.push(text.clone());
                    self.sender
                        .send(HandlerToLooperMessage::Assistant(text))
                        .await?;
                }
                Ok(ResponseStreamEvent::ResponseReasoningSummaryTextDelta(delta)) => {
                    let text = delta.delta.clone();
                    self.sender
                        .send(HandlerToLooperMessage::Thinking(text))
                        .await?;
                }
                Ok(ResponseStreamEvent::ResponseReasoningSummaryTextDone(_)) => {
                    self.sender
                        .send(HandlerToLooperMessage::ThinkingComplete)
                        .await?;
                }
                Ok(ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(delta)) => {
                    self.sender
                        .send(HandlerToLooperMessage::ToolCallPending(
                            delta.item_id.clone(),
                        ))
                        .await?;
                }
                Ok(ResponseStreamEvent::ResponseOutputItemDone(item_done)) => {
                    if let OutputItem::FunctionCall(fc) = item_done.item {
                        function_calls.push(fc);
                    }
                }
                Ok(ResponseStreamEvent::ResponseCompleted(completed)) => {
                    response_id = Some(completed.response.id.clone());
                }
                Ok(_) => {}
                Err(err) => {
                    println!("error: {err:?}");
                }
            }
        }

        if let Some(id) = response_id {
            self.previous_response_id = Some(id);
        }

        if !function_calls.is_empty() {
            let mut input_items: Vec<InputItem> = Vec::new();

            if function_calls
                .iter()
                .any(|call| is_ask_user_question_tool(&call.name))
            {
                if function_calls.len() > 1 {
                    for fc in function_calls {
                        let tcr = HandlerToLooperToolCallRequest {
                            id: fc.call_id.clone(),
                            name: fc.name.clone(),
                            args: serde_json::from_str(&fc.arguments).unwrap_or_default(),
                        };

                        emit_tool_call_request(&self.sender, &tcr).await?;
                        emit_tool_call_complete(&self.sender, tcr.id.clone()).await?;

                        input_items.push(InputItem::Item(Item::FunctionCallOutput(
                            FunctionCallOutputItemParam {
                                call_id: tcr.id,
                                output: FunctionCallOutput::Text(
                                    ask_user_question_batch_error().to_string(),
                                ),
                                id: None,
                                status: None,
                            },
                        )));
                    }
                } else {
                    let fc = function_calls.into_iter().next().unwrap();
                    let tcr = HandlerToLooperToolCallRequest {
                        id: fc.call_id.clone(),
                        name: fc.name.clone(),
                        args: serde_json::from_str(&fc.arguments).unwrap_or_default(),
                    };

                    let value =
                        await_user_input(&self.sender, &self.user_response_receiver, &tcr).await?;

                    emit_tool_call_complete(&self.sender, tcr.id.clone()).await?;

                    input_items.push(InputItem::Item(Item::FunctionCallOutput(
                        FunctionCallOutputItemParam {
                            call_id: tcr.id,
                            output: FunctionCallOutput::Text(value.to_string()),
                            id: None,
                            status: None,
                        },
                    )));
                }

                return self
                    .inner_send_message(Some(InputParam::Items(input_items)), tools_runner)
                    .await;
            }

            let mut tool_join_set = JoinSet::new();

            for fc in function_calls {
                let tcr = HandlerToLooperToolCallRequest {
                    id: fc.call_id.clone(),
                    name: fc.name.clone(),
                    args: serde_json::from_str(&fc.arguments).unwrap_or_default(),
                };

                emit_tool_call_request(&self.sender, &tcr).await?;

                let tr = tools_runner.clone();
                let fc_clone = fc.clone();
                tool_join_set.spawn(async move {
                    let args: Value = serde_json::from_str(&fc_clone.arguments).unwrap_or_default();
                    let result = tr.run_tool(fc_clone.name.clone(), args).await;
                    (fc_clone.call_id.clone(), result)
                });
            }

            while let Some(result) = tool_join_set.join_next().await {
                match result {
                    Ok((call_id, value)) => {
                        emit_tool_call_complete(&self.sender, call_id.clone()).await?;

                        input_items.push(InputItem::Item(Item::FunctionCallOutput(
                            FunctionCallOutputItemParam {
                                call_id,
                                output: FunctionCallOutput::Text(value.to_string()),
                                id: None,
                                status: None,
                            },
                        )));
                    }
                    Err(e) => {
                        eprintln!(
                            "Join Error occured when collecting tool call results | Error: {}",
                            e
                        );
                    }
                }
            }

            return self
                .inner_send_message(Some(InputParam::Items(input_items)), tools_runner)
                .await;
        }

        Ok(assistant_res_buf.join(""))
    }
}

#[async_trait]
impl StreamingChatHandler for OpenAIResponsesHandler {
    async fn send_message(
        &mut self,
        message_history: Option<MessageHistory>,
        message: &str,
        tools_runner: Arc<dyn LooperTools>,
    ) -> Result<MessageHistory> {
        if let Some(MessageHistory::ResponseId(id)) = message_history {
            self.previous_response_id = Some(id);
        }

        let input = InputParam::Text(message.to_string());
        self.inner_send_message(Some(input), tools_runner).await?;

        self.sender
            .send(HandlerToLooperMessage::TurnComplete)
            .await?;

        Ok(MessageHistory::ResponseId(
            self.previous_response_id.clone().unwrap_or_default(),
        ))
    }

    fn set_tools(&mut self, tools: Vec<LooperToolDefinition>) {
        self.tools = tools
            .into_iter()
            .map(|t| Tool::Function(t.into()))
            .collect();
    }
}
