"""Default system prompts theo LLM backend — nguồn sự thật duy nhất.

Handler dùng khi system_prompt=None; server đọc để trả "effective prompt"
cho UI hiển thị prompt thực tế model sẽ dùng.
"""

# LƯU Ý Qwen3-4B: prompt dài hơn (kèm quy tắc ngắn gọn/emoji/không bịa) làm model
# KHÔNG gọi tool — chỉ bản tối giản này gọi ổn định. Model mạnh hơn có thể nới.
TRANSFORMERS_PROMPT = (
    "Bạn là trợ lý tiếng Việt. "
    "Khi câu hỏi liên quan tài liệu, giá cả, chính sách nội bộ: "
    "gọi tool search_knowledge trước."
)

OPENAI_COMPATIBLE_PROMPT = (
    "Bạn là trợ lý giọng nói tiếng Việt. "
    "Trả lời ngắn gọn, tự nhiên, phù hợp hội thoại nói. "
    "Khi câu hỏi liên quan tài liệu, chính sách, báo cáo, số liệu nội bộ: "
    "hãy gọi tool search_knowledge trước khi trả lời."
)

# backend → default prompt (transformers riêng; openai/hf-router/gemini/local
# đều là OpenAI-compatible family)
DEFAULT_SYSTEM_PROMPTS: dict[str, str] = {
    "transformers": TRANSFORMERS_PROMPT,
    "openai": OPENAI_COMPATIBLE_PROMPT,
    "hf-router": OPENAI_COMPATIBLE_PROMPT,
    "gemini": OPENAI_COMPATIBLE_PROMPT,
    "local": OPENAI_COMPATIBLE_PROMPT,
}


def effective_system_prompt(backend: str, custom: str | None) -> str:
    """Prompt thực tế sẽ dùng: custom nếu set, ngược lại default của backend."""
    return custom if custom else DEFAULT_SYSTEM_PROMPTS.get(backend, OPENAI_COMPATIBLE_PROMPT)
