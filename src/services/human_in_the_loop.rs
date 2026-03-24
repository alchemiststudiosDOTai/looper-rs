use anyhow::{Result, anyhow, bail};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::{
    Mutex,
    mpsc::{Receiver, Sender},
};

use crate::{
    tools::to_user_input_request,
    types::{
        HandlerToLooperMessage, HandlerToLooperToolCallRequest, LooperToHandlerToolCallResult,
    },
};

pub async fn emit_tool_call_request(
    sender: &Sender<HandlerToLooperMessage>,
    call: &HandlerToLooperToolCallRequest,
) -> Result<()> {
    sender
        .send(HandlerToLooperMessage::ToolCallRequest(call.clone()))
        .await?;
    Ok(())
}

pub async fn emit_tool_call_complete(
    sender: &Sender<HandlerToLooperMessage>,
    id: String,
) -> Result<()> {
    sender
        .send(HandlerToLooperMessage::ToolCallComplete(id))
        .await?;
    Ok(())
}

pub async fn await_user_input(
    sender: &Sender<HandlerToLooperMessage>,
    receiver: &Option<Arc<Mutex<Receiver<LooperToHandlerToolCallResult>>>>,
    call: &HandlerToLooperToolCallRequest,
) -> Result<Value> {
    emit_tool_call_request(sender, call).await?;

    sender
        .send(HandlerToLooperMessage::UserInputRequest(
            to_user_input_request(call),
        ))
        .await?;

    let rx = receiver.as_ref().ok_or_else(|| {
        anyhow!(
            "ask_user_question was called, but no user_response_receiver was configured on the builder"
        )
    })?;

    let mut rx = rx.lock().await;

    let response = rx.recv().await.ok_or_else(|| {
        anyhow!("ask_user_question response channel closed before the user replied")
    })?;

    if !response.id.is_empty() && response.id != call.id {
        bail!(
            "ask_user_question received response for unexpected tool call id: expected {}, got {}",
            call.id,
            response.id
        );
    }

    Ok(response.value)
}
