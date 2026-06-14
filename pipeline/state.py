from __future__ import annotations

from typing import Any, Optional, TypedDict


class Pipeline_State(TypedDict, total=False):
    input_file: str
    output_dir: str

    file_hash: str
    cache_hit: bool

    session_id: str
    masked_filename: str

    original_text: str
    masked_text: str
    unmasked_json: dict[str, Any]

    # We will store the extracted chunks here
    chunk: list[dict[str, str]]

    # --- LLM Specific State Additions ---
    user_instruction: str
    json_schema: dict[str, Any]
    extracted_json: dict[str, Any]

    error: Optional[str]
