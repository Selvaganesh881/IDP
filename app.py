import asyncio
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Import your DB setup (Assuming you put your schema initialization in db/schema.py)
from masking.db.connection import get_db
from masking.db.schema import initialize_schema

# Import your LangGraph builder
from pipeline.graph import build_pipeline_graph

# --- Application Startup & DB Init ---
Path("data").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)


async def init_database():
    async with get_db() as db:
        await initialize_schema(db)


# Run DB init once when the app starts
try:
    asyncio.run(init_database())
except RuntimeError:
    # Streamlit sometimes gets finicky with existing event loops; this safely handles it
    pass

# --- Streamlit Page Config ---
st.set_page_config(page_title="Financial IDP PoC (LangGraph Engine)", layout="wide")

DEFAULT_INSTRUCTION = """
    Extract the parameters defined in the schema from the document provided below.

    EXTRACTION INSTRUCTIONS:
    - Extract values exactly as they appear in the text.
    - If a specific value cannot be found in the document, you must still include the key in your JSON response.
    - For missing string values, use the exact text: "NOT_FOUND".
    - For missing numbers or booleans, use: null.
    
    """
DEFAULT_SCHEMA = """{
  "title": "DocumentExtraction",
  "type": "object",
  "properties": {
    "account_holder_name": {"type": "string"},
    "total_balance": {"type": "number"},
    "account_number": {"type": "string"}
  },
  "required": ["account_holder_name", "total_balance"]
}"""

st.title("Document AI: Field Extraction with PII Faking & LLMs")

tab_dash, tab_template, tab_results = st.tabs(
    ["📊 Dashboard", "⚙️ Template & Process", "🗄️ Result Database"]
)

with tab_template:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("1. Configuration")
        uploaded_file = st.file_uploader("Upload Financial PDF", type=["pdf"])
        user_instruction = st.text_area(
            "User Instruction", value=DEFAULT_INSTRUCTION, height=100
        )
        json_schema_input = st.text_area(
            "Expected JSON Schema", value=DEFAULT_SCHEMA, height=300
        )
        process_btn = st.button(
            "🚀 Execute LangGraph Pipeline", use_container_width=True, type="primary"
        )

    with col_right:
        st.subheader("2. Pipeline Execution")

        if process_btn and uploaded_file is not None:
            try:
                parsed_schema = json.loads(json_schema_input)
                file_path = f"uploads/{uploaded_file.name}"

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.status("Executing StateGraph...", expanded=True) as status:
                    st.write("Initializing LangGraph Nodes...")

                    # 1. Build the compiled LangGraph application
                    graph_app = build_pipeline_graph()

                    # 2. Define the initial state payload
                    initial_state = {
                        "input_file": file_path,
                        "user_instruction": user_instruction,
                        "json_schema": parsed_schema,
                    }

                    # 3. Execute the graph asynchronously
                    st.write(
                        "Traversing nodes (Check Cache -> Ingest -> Mask -> Extract)..."
                    )
                    final_state = asyncio.run(graph_app.ainvoke(initial_state))

                    status.update(
                        label="Graph Execution Complete!",
                        state="complete",
                        expanded=False,
                    )

                # --- Result Rendering ---
                if "error" in final_state:
                    st.error(f"Pipeline Failed: {final_state['error']}")
                else:
                    if final_state.get("cache_hit"):
                        st.success(
                            "✅ Cache Hit! Retrieved existing processing session."
                        )

                    st.divider()
                    st.subheader("🔍 Document Comparison")

                    # Create two side-by-side columns
                    col_orig, col_faked = st.columns(2)

                    with col_orig:
                        st.caption("📄 Original Raw Markdown")
                        orig_txt = final_state.get("original_text", "")
                        # Show first 2000 chars to prevent massive UI scrolling
                        st.code(
                            orig_txt[:2000] + "\n\n...[TRUNCATED]", language="markdown"
                        )

                    with col_faked:
                        st.caption("🛡️ Anonymized Markdown (Sent to Qwen)")
                        fake_txt = final_state.get("masked_text", "")
                        st.code(
                            fake_txt[:2000] + "\n\n...[TRUNCATED]", language="markdown"
                        )

                    st.divider()
                    st.subheader("🧠 Extraction Results")

                    col_json_fake, col_json_real = st.columns(2)

                    with col_json_fake:
                        st.caption("🔒 LLM Output (Anonymized Data)")
                        st.json(final_state.get("extracted_json", {}))

                    with col_json_real:
                        st.caption("🔓 Final Output (Restored Real Data)")
                        st.json(final_state.get("unmasked_json", {}))

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
        elif process_btn:
            st.warning("Please upload a PDF document first.")

# Tab 3 (Results) remains mostly the same, querying your aiosqlite DB instead of the standard sqlite3.
# I recommend keeping Tab 3 simple for the PoC or using pandas to read the sqlite file directly for rendering.
