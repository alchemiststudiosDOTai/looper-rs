use std::fmt;

use tokio::sync::{mpsc, oneshot};

pub type AskUserSender = mpsc::Sender<AskUserRequest>;

pub struct AskUserRequest {
    pub id: String,
    pub question: String,
    pub options: Vec<String>,
    pub response_tx: oneshot::Sender<AskUserResponse>,
}

impl fmt::Debug for AskUserRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AskUserRequest")
            .field("id", &self.id)
            .field("question", &self.question)
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct AskUserResponse {
    pub answer: String,
}
