pub type Model<'a> = &'a str;

pub enum Handlers<'a> {
    OpenAICompletions(Model<'a>),
    OpenAIResponses(Model<'a>),
    Anthropic(Model<'a>)
}
