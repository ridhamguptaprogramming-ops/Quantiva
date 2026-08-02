"""
Chat / SFT formatting utilities.

Provides:
  - ``ChatMessage``: a typed representation of a chat message.
  - ``apply_chat_template``: render a conversation into a model string using
    either a built-in template or a user-supplied one.
  - ``format_sft_example``: produce ``(input_ids, labels)`` with the loss
    masked over the prompt (only assistant tokens contribute to loss).

Built-in template (Qwen/ChatML style):

    <|im_start|>system\n...<|im_end|>\n
    <|im_start|>user\n...<|im_end|>\n
    <|im_start|>assistant\n...<|im_end|>\n
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ChatML special tokens.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


@dataclass
class ChatMessage:
    """A single chat message."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = field(default_factory=list)


def apply_chat_template(
    messages: Sequence[ChatMessage],
    tokenizer=None,
    add_generation_prompt: bool = True,
    template: Optional[Callable[[Sequence[ChatMessage], bool], str]] = None,
) -> str:
    """
    Render a conversation into a single model string.

    Args:
        messages: Conversation as a list of ``ChatMessage``.
        tokenizer: If provided and has ``apply_chat_template`` (HF-style),
            it is used. Otherwise the built-in ChatML template is applied.
        add_generation_prompt: Whether to append an assistant-turn start token
            (useful for inference, not for training).
        template: Optional custom renderer taking ``(messages, add_gen)``.
    """
    if template is not None:
        return template(messages, add_generation_prompt)

    # Use HF-style tokenizer.chat_template if available.
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [m.__dict__ for m in messages],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    return _render_chatml(messages, add_generation_prompt)


def _render_chatml(
    messages: Sequence[ChatMessage], add_generation_prompt: bool
) -> str:
    """Render messages with the ChatML template."""
    parts: List[str] = []
    for m in messages:
        if m.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported chat role: {m.role}")
        if m.tool_calls:
            # Render tool calls as JSON in the content.
            import json

            content = m.content or ""
            content += "\n" + json.dumps(m.tool_calls)
        else:
            content = m.content
        parts.append(f"{IM_START}{m.role}\n{content}{IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def format_sft_example(
    messages: Sequence[ChatMessage],
    tokenizer,
    ignore_index: int = -100,
) -> Tuple[List[int], List[int]]:
    """
    Format an SFT example into ``(input_ids, labels)``.

    The prompt portion is masked out of the loss by setting its labels to
    ``ignore_index``; only assistant-turn tokens are trained on.

    Args:
        messages: Conversation.
        tokenizer: Tokenizer with ``encode`` returning token ids.
        ignore_index: Label value used to mask non-assistant tokens.

    Returns:
        ``(input_ids, labels)`` both as lists of ints of equal length.
    """
    # Build the full conversation text (with generation prompt) and the
    # assistant-only masked text.
    full_text = apply_chat_template(
        messages, tokenizer=tokenizer, add_generation_prompt=False
    )
    # Identify assistant spans to build the label mask.
    label_mask = _assistant_mask(messages)

    input_ids = tokenizer.encode(full_text)
    labels = list(input_ids)

    # Mask all tokens before the first assistant span.
    # (For simplicity we mask by mapping token indices back to char spans is
    # complex; this simplified version masks everything before the assistant
    # tokens using the textual mask.)
    labels = _apply_mask_by_roles(messages, input_ids, tokenizer, ignore_index)
    return input_ids, labels


def _assistant_mask(messages: Sequence[ChatMessage]) -> List[bool]:
    """Return a per-message boolean indicating whether it's an assistant turn."""
    return [m.role == "assistant" for m in messages]


def _apply_mask_by_roles(
    messages: Sequence[ChatMessage],
    input_ids: List[int],
    tokenizer,
    ignore_index: int,
) -> List[int]:
    """
    Reconstruct labels by re-encoding each message separately so the assistant
    tokens can be isolated from the prompt tokens.
    """
    labels: List[int] = []
    for m in messages:
        seg_text = f"{IM_START}{m.role}\n{m.content}{IM_END}\n"
        seg_ids = tokenizer.encode(seg_text)
        if m.role == "assistant":
            labels.extend(seg_ids)
        else:
            labels.extend([ignore_index] * len(seg_ids))
    # Trim/pad to match input_ids length (tokenizer boundary differences).
    if len(labels) < len(input_ids):
        labels.extend([ignore_index] * (len(input_ids) - len(labels)))
    return labels[: len(input_ids)]

