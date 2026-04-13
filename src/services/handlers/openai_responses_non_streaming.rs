use std::sync::Arc;

use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        chat::ReasoningEffort,
        responses::{
            CreateResponseArgs, FunctionCallOutput, FunctionCallOutputItemParam, InputItem,
            InputParam, Item, OutputItem, Reasoning, ReasoningSummary, Tool,
        },
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

pub struct OpenAIResponsesNonStreamingHandler {
    client: Client<OpenAIConfig>,
    model: String,
    previous_response_id: Option<String>,
    tools: Vec<Tool>,
    instructions: String,
}

impl OpenAIResponsesNonStreamingHandler {
    pub fn new(model: &str, system_message: &str) -> Result<Self> {
        let client = Client::new();

        Ok(OpenAIResponsesNonStreamingHandler {
            client,
            model: model.to_string(),
            previous_response_id: None,
            tools: Vec::new(),
            instructions: system_message.to_string(),
        })
    }

    #[async_recursion]
    async fn inner_send_message(
        &mut self,
        input: Option<InputParam>,
        tools_runner: Arc<dyn LooperTools>,
        steps: &mut Vec<TurnStep>,
    ) -> Result<()> {
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
        let response = self.client.responses().create(request).await?;

        self.previous_response_id = Some(response.id.clone());

        let mut thinking = Vec::new();
        let mut text = None;
        let mut function_calls = Vec::new();

        for item in &response.output {
            match item {
                OutputItem::Reasoning(r) => {
                    for part in &r.summary {
                        let async_openai::types::responses::SummaryPart::SummaryText(s) = part;
                        thinking.push(ThinkingBlock {
                            content: s.text.clone(),
                        });
                    }
                }
                OutputItem::Message(m) => {
                    for content in &m.content {
                        if let async_openai::types::responses::OutputMessageContent::OutputText(t) =
                            content
                        {
                            text = Some(t.text.clone());
                        }
                    }
                }
                OutputItem::FunctionCall(fc) => {
                    function_calls.push(fc.clone());
                }
                _ => {}
            }
        }

        let mut tool_call_records = Vec::new();

        if !function_calls.is_empty() {
            let exclusive_names = exclusive_tool_names(
                tools_runner.as_ref(),
                function_calls.iter().map(|call| call.name.as_str()),
            );
            let mut input_items: Vec<InputItem> = Vec::new();

            if !exclusive_names.is_empty() && function_calls.len() > 1 {
                let error_result = invalid_exclusive_tool_batch_result(&exclusive_names);

                for fc in function_calls {
                    let args: serde_json::Value =
                        serde_json::from_str(&fc.arguments).unwrap_or_default();

                    tool_call_records.push(ToolCallRecord {
                        id: fc.call_id.clone(),
                        name: fc.name.clone(),
                        args,
                        result: error_result.clone(),
                    });

                    input_items.push(InputItem::Item(Item::FunctionCallOutput(
                        FunctionCallOutputItemParam {
                            call_id: fc.call_id,
                            output: FunctionCallOutput::Text(error_result.to_string()),
                            id: None,
                            status: None,
                        },
                    )));
                }
            } else if !exclusive_names.is_empty() {
                let fc = function_calls.into_iter().next().expect("function call missing");
                let args: serde_json::Value =
                    serde_json::from_str(&fc.arguments).unwrap_or_default();
                let result = tools_runner.run_tool(fc.name.clone(), args.clone()).await;

                tool_call_records.push(ToolCallRecord {
                    id: fc.call_id.clone(),
                    name: fc.name.clone(),
                    args,
                    result: result.clone(),
                });

                input_items.push(InputItem::Item(Item::FunctionCallOutput(
                    FunctionCallOutputItemParam {
                        call_id: fc.call_id,
                        output: FunctionCallOutput::Text(result.to_string()),
                        id: None,
                        status: None,
                    },
                )));
            } else {
                let mut tool_join_set = JoinSet::new();
                let mut ordered_results = Vec::new();
                ordered_results.resize_with(function_calls.len(), || None);

                for (index, fc) in function_calls.into_iter().enumerate() {
                    let tr = tools_runner.clone();
                    tool_join_set.spawn(async move {
                        let args: serde_json::Value =
                            serde_json::from_str(&fc.arguments).unwrap_or_default();
                        let result = tr.run_tool(fc.name.clone(), args.clone()).await;

                        (index, result, fc, args)
                    });
                }

                while let Some(result) = tool_join_set.join_next().await {
                    match result {
                        Ok((index, result, fc, args)) => {
                            ordered_results[index] = Some((result, fc, args));
                        }
                        Err(e) => {
                            eprintln!(
                                "Join Error occured when collecting tool call results | Error: {}",
                                e
                            );
                        }
                    }
                }

                for (result, fc, args) in ordered_results.into_iter().flatten() {
                    tool_call_records.push(ToolCallRecord {
                        id: fc.call_id.clone(),
                        name: fc.name.clone(),
                        args,
                        result: result.clone(),
                    });

                    input_items.push(InputItem::Item(Item::FunctionCallOutput(
                        FunctionCallOutputItemParam {
                            call_id: fc.call_id,
                            output: FunctionCallOutput::Text(result.to_string()),
                            id: None,
                            status: None,
                        },
                    )));
                }
            }

            steps.push(TurnStep {
                thinking,
                text,
                tool_calls: tool_call_records,
            });

            return self
                .inner_send_message(Some(InputParam::Items(input_items)), tools_runner, steps)
                .await;
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
impl ChatHandler for OpenAIResponsesNonStreamingHandler {
    async fn send_message(
        &mut self,
        message_history: Option<MessageHistory>,
        message: &str,
        tools_runner: Arc<dyn LooperTools>,
    ) -> Result<TurnResult> {
        if let Some(MessageHistory::ResponseId(id)) = message_history {
            self.previous_response_id = Some(id);
        }

        let input = InputParam::Text(message.to_string());

        let mut steps = Vec::new();
        self.inner_send_message(Some(input), tools_runner, &mut steps)
            .await?;

        let final_text = steps.iter().rev().find_map(|s| s.text.clone());

        let message_history =
            MessageHistory::ResponseId(self.previous_response_id.clone().unwrap_or_default());

        Ok(TurnResult {
            steps,
            final_text,
            message_history,
        })
    }

    fn set_tools(&mut self, tools: Vec<LooperToolDefinition>) {
        self.tools = tools
            .into_iter()
            .map(|t| Tool::Function(t.into()))
            .collect();
    }
}
