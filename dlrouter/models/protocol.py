"""
OpenAI-compatible protocol models for DLRouter.
Modified from https://github.com/InternLM/lmdeploy/blob/133dccb6f96ca4fb41eec3854f31aead9e07050b/lmdeploy/serve/openai/protocol.py.
"""

import time
from typing import Any, Literal, Optional, Union

import shortuuid
from pydantic import BaseModel, Field


# ---- Model listing ----


class ModelPermission(BaseModel):
    """Model permission."""

    id: str = Field(default_factory=lambda: f'modelperm-{shortuuid.random()}')
    object: str = 'model_permission'
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = True
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = '*'
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    """Model card."""

    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'dlrouter'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: list[ModelPermission] = Field(default_factory=list)


class ModelList(BaseModel):
    """Model list."""

    object: str = 'list'
    data: list[ModelCard] = Field(default_factory=list)


# ---- Chat completions ----


class SessionParams(BaseModel):
    """Session parameters for consistent routing."""

    session_id: Optional[str] = None


class StreamOptions(BaseModel):
    """Stream options."""

    include_usage: Optional[bool] = False


class JsonSchema(BaseModel):
    """JSON schema."""

    name: str
    schema_: Optional[dict[str, Any]] = Field(default=None, alias='schema')


class ResponseFormat(BaseModel):
    """Response format."""

    type: Literal['text', 'json_object', 'json_schema', 'regex_schema']
    json_schema: Optional[JsonSchema] = None
    regex_schema: Optional[str] = None


class FunctionDefinition(BaseModel):
    """Function definition."""

    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class Tool(BaseModel):
    """Tool definition."""

    type: str = 'function'
    function: FunctionDefinition


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""

    name: str


class ToolChoice(BaseModel):
    """The tool choice definition."""

    function: ToolChoiceFuncName
    type: Literal['function'] = Field(default='function', examples=['function'])


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI-compatible)."""

    model: str
    messages: Union[str, list[dict[str, Any]]] = Field(examples=[[{'role': 'user', 'content': 'hi'}]])
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 40
    n: Optional[int] = 1
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0
    user: Optional[str] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Union[ToolChoice, Literal['auto', 'required', 'none']] = 'auto'
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    logit_bias: Optional[dict[str, float]] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    # Extended fields (backend specific, pass through)
    extra_body: Optional[dict[str, Any]] = Field(default=None, description='Extra fields forwarded to backend')
    # Routing fields
    session_params: Optional[SessionParams] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    ignore_eos: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    return_token_ids: Optional[bool] = False


class FunctionCall(BaseModel):
    """Function response."""

    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call response."""

    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    type: Literal['function'] = 'function'
    function: FunctionCall


class ChatMessage(BaseModel):
    """Chat message."""

    role: str
    content: Optional[str] = None
    gen_tokens: Optional[list[int]] = None
    reasoning_content: Optional[str] = Field(default=None, examples=[None])
    tool_calls: Optional[list[ToolCall]] = Field(default=None, examples=[None])


class DeltaFunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


# a tool call delta where everything is optional
class DeltaToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f'chatcmpl-tool-{shortuuid.random()}')
    type: Literal['function'] = 'function'
    index: int
    function: Optional[DeltaFunctionCall] = None


class DeltaMessage(BaseModel):
    """Delta message for streaming."""

    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    gen_tokens: Optional[list[int]] = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LogProbs(BaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[Optional[float]] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: Optional[list[Optional[dict[str, float]]]] = None


class TopLogprob(BaseModel):
    token: str
    bytes: Optional[list[int]] = None
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: Optional[list[int]] = None
    logprob: float
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(BaseModel):
    content: Optional[list[ChatCompletionTokenLogprob]] = None


class ChatCompletionResponseChoice(BaseModel):
    """Chat completion response choices."""

    index: int
    message: ChatMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error']] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class ChatCompletionStreamChoice(BaseModel):
    """Stream choice."""

    index: int
    delta: DeltaMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error']] = None


class ChatCompletionStreamResponse(BaseModel):
    """Chat completion stream response."""

    id: str = Field(default_factory=lambda: f'chatcmpl-{shortuuid.random()}')
    object: str = 'chat.completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ''
    choices: list[ChatCompletionStreamChoice] = Field(default_factory=list)
    usage: Optional[UsageInfo] = None


# ---- Text completions ----


class CompletionRequest(BaseModel):
    """Completion request."""

    model: str
    prompt: Union[str, list[Any]]
    suffix: Optional[str] = None
    temperature: Optional[float] = 0.7
    n: Optional[int] = 1
    logprobs: Optional[int] = None
    max_completion_tokens: Optional[int] = Field(
        default=None,
        examples=[None],
        description=(
            'An upper bound for the number of tokens that can be generated for a completion, '
            'including visible output tokens and reasoning tokens'
        ),
    )
    max_tokens: Optional[int] = Field(
        default=16,
        examples=[16],
        deprecated='max_tokens is deprecated in favor of the max_completion_tokens field',
    )
    stop: Optional[Union[str, list[str]]] = Field(default=None, examples=[None])
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = Field(default=None, examples=[None])
    top_p: Optional[float] = 1.0
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
    # additional argument of lmdeploy
    repetition_penalty: Optional[float] = 1.0
    session_id: Optional[int] = -1
    ignore_eos: Optional[bool] = False
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    top_k: Optional[int] = 40  # for opencompass
    seed: Optional[int] = None
    min_p: float = 0.0
    return_token_ids: Optional[bool] = False
    # Routing fields
    session_params: Optional[SessionParams] = None
    user_id: Optional[str] = None


class CompletionResponseChoice(BaseModel):
    """Completion response choices."""

    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    gen_tokens: Optional[list[int]] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error', 'abort']] = None


class CompletionResponse(BaseModel):
    """Completion response."""

    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    """Completion response stream choice."""

    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    gen_tokens: Optional[list[int]] = None
    finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'error', 'abort']] = None


class CompletionStreamResponse(BaseModel):
    """Completion stream response."""

    id: str = Field(default_factory=lambda: f'cmpl-{shortuuid.random()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = None
