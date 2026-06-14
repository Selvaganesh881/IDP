import asyncio
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from masking.db import repository
from masking.db.connection import get_db
from masking.db.schema import initialize_schema
from pipeline.graph import build_pipeline_graph

# --- App initialization ---
Path("data").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)


def run_async(coro: Any) -> Any:
    """Run an async coroutine safely from Streamlit."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


async def init_database() -> None:
    async with get_db() as db:
        await initialize_schema(db)


run_async(init_database())

# --- Streamlit page configuration ---
st.set_page_config(
    page_title="Document AI",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #f2f6fc 100%);
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 0.65rem;
    }
    .stButton>button:hover {
        background-color: #0747c7;
        color: white;
    }
    .app-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .app-icon {
        width: 48px;
        height: 48px;
        display: grid;
        place-items: center;
        font-size: 1.25rem;
        background: #eef4ff;
        border-radius: 14px;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #475569;
        margin: 0;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #0f172a;
    }
    .panel {
        padding: 1rem;
        border-radius: 1rem;
        background: #ffffffee;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }
    .big-number {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0b3c78;
    }
    .metric-label {
        color: #334155;
        margin-bottom: 0.2rem;
    }
    .card {
        border-radius: 1rem;
        padding: 1.3rem;
        background: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    }
    .panel {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 1rem;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .panel-header {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        align-items: center;
        margin-bottom: 1rem;
    }
    .panel-title {
        font-size: 1rem;
        font-weight: 700;
        color: #0f172a;
    }
    .panel-label {
        color: #475569;
        font-size: 0.95rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #0f172a;
    }
    .section-subtitle {
        color: #475569;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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


@st.cache_resource
def get_graph_app():
    return build_pipeline_graph()


def fetch_all_sessions() -> list[dict[str, Any]]:
    try:

        async def _query():
            async with get_db() as db:
                return await repository.get_all_session(db)

        return run_async(_query())
    except Exception:
        return []


def render_kpi_card(title: str, value: Any, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class='card'>
            <div class='metric-label'>{title}</div>
            <div class='big-number'>{value}</div>
            <div>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_parse_schema(schema_text: str) -> Any:
    try:
        return json.loads(schema_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON schema: {exc}") from exc


def pretty_json(value: Any) -> Any:
    if value is None:
        return {}
    return value


st.markdown(
    """
    <div class='app-header'>
        <div class='app-icon'>🧾</div>
        <div>
            <div class='app-title'>Document AI</div>
            <div class='subtitle'>Field Extraction with PII Faking & LLM</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

session_runs = fetch_all_sessions()

st.write("---")

tab_template, tab_dash, tab_insights = st.tabs(
    ["⚙️ Template & Process", "📊 Dashboard", "🔍 Insights"]
)

with tab_template:
    st.markdown(
        "<div class='section-title'>1. Configuration</div>", unsafe_allow_html=True
    )
    st.markdown(
        "<div class='section-subtitle'>Upload a PDF, refine the extraction schema, and start the secure PII-aware extraction pipeline.</div>",
        unsafe_allow_html=True,
    )

    with st.form("process_form", clear_on_submit=False):
        col_inputs, col_info = st.columns([2, 1])

        with col_inputs:
            uploaded_file = st.file_uploader("Upload Financial PDF", type=["pdf"])
            user_instruction = st.text_area(
                "LLM Instruction",
                value=DEFAULT_INSTRUCTION,
                height=140,
                help="Describe how the model should extract structured values from the document.",
            )
            json_schema_input = st.text_area(
                "Extraction JSON Schema",
                value=DEFAULT_SCHEMA,
                height=240,
                help="Define the exact schema you expect from the LLM output.",
            )

            process_btn = st.form_submit_button("🚀 Run Extraction")

        with col_info:
            st.markdown("### Demo tips")
            st.markdown(
                """
                - Upload one PDF at a time.
                - Use the sample schema and adjust only when necessary.
                - The pipeline preserves a secure audit trail of each run.
                """
            )

    if process_btn:
        if uploaded_file is None:
            st.warning("Please upload a PDF document before running the pipeline.")
        else:
            try:
                parsed_schema = safe_parse_schema(json_schema_input)
                file_path = Path("uploads") / uploaded_file.name
                file_path.write_bytes(uploaded_file.getbuffer())

                graph_app = get_graph_app()
                initial_state = {
                    "input_file": str(file_path),
                    "user_instruction": user_instruction,
                    "json_schema": parsed_schema,
                }

                with st.spinner("Executing pipeline and extracting values..."):
                    final_state = run_async(graph_app.ainvoke(initial_state))

                if "error" in final_state:
                    st.error(f"Pipeline error: {final_state['error']}")
                else:
                    if final_state.get("cache_hit"):
                        st.success("✅ Cache hit: existing session reused.")

                    st.success("Pipeline completed successfully.")
                    st.write("---")

                    st.subheader("Document comparison")
                    col_orig, col_faked = st.columns(2)

                    with col_orig:
                        st.markdown("**Original extracted text**")
                        orig_txt = final_state.get("original_text", "")
                        st.text_area(
                            "Original text",
                            value=orig_txt,
                            height=250,
                            key="orig_text",
                        )

                    with col_faked:
                        st.markdown("**Anonymized text sent to the LLM**")
                        masked_text = final_state.get("masked_text", "")
                        st.text_area(
                            "Anonymized text",
                            value=masked_text,
                            height=250,
                            key="masked_text",
                        )

                    st.write("---")
                    st.subheader("Extraction results")
                    result_cols = st.columns(2)
                    with result_cols[0]:
                        st.markdown("**LLM output (anonymized)**")
                        st.json(pretty_json(final_state.get("extracted_json")))
                    with result_cols[1]:
                        st.markdown("**Final output (unmasked)**")
                        st.json(pretty_json(final_state.get("unmasked_json")))

                    st.write("---")
                    st.info(
                        "The final output is restored to real values while keeping the extraction process and audit trail secure."
                    )
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

with tab_dash:
    st.subheader("Executive snapshot")
    st.markdown(
        """
        - **Business value:** reduce manual review time while preserving sensitive information.
        - **Security posture:** PII is masked before LLM extraction and restored only in the final result.
        - **Audit readiness:** every run is recorded with document metadata and entity counts.
        """
    )

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Documents Processed", len(session_runs), "Total historical runs")
    kpi2.metric(
        "PII Entities Masked",
        sum(session.get("entity_count", 0) for session in session_runs),
        "Sensitive values anonymized",
    )
    average_entities = (
        sum(session.get("entity_count", 0) for session in session_runs)
        / len(session_runs)
        if session_runs
        else 0
    )
    kpi3.metric(
        "Avg PII Entities per Doc",
        f"{average_entities:.1f}",
        "Average anonymized values per document",
    )

    st.write("---")
    st.subheader("Key insights")
    if session_runs:
        latest_runs = session_runs[:3]
        for run in latest_runs:
            st.markdown(
                f"- **{run['original_filename']}** — {run['entity_count']} PII values masked on {run['created_at']}"
            )
        st.markdown(
            """
            - The pipeline checks file hashes to prevent duplicate processing.
            - Masking occurs before LLM extraction, ensuring sensitive text is never sent in raw form.
            - Final JSON results are unmasked only after structured extraction completes.
            """
        )
    else:
        st.info(
            "No processing insights available yet. Run the extraction pipeline first."
        )

with tab_insights:
    st.subheader("Insights & Trends")
    total_documents = len(session_runs)
    total_entities = sum(session.get("entity_count", 0) for session in session_runs)
    avg_entities = total_entities / total_documents if total_documents else 0

    insight_cols = st.columns(3)
    insight_cols[0].metric("Total Documents", total_documents)
    insight_cols[1].metric("Total PII Entities", total_entities)
    insight_cols[2].metric("Average Entities", f"{avg_entities:.1f}")

    st.markdown("### Operational insights")
    st.markdown(
        """
        - **Secure extract-first design:** text is masked before LLM processing.
        - **Schema-driven output:** the pipeline follows the provided JSON schema for structured extraction.
        - **Enterprise readiness:** proof-of-concept built for clear audit trails and executive review.
        """
    )

    if session_runs:
        st.markdown("### Latest run summaries")
        for run in session_runs[:5]:
            st.markdown(
                f"- `{run['session_id'][:8]}...` • {run['original_filename']} • {run['entity_count']} entities • {run['created_at']}"
            )
    else:
        st.info("No recent run summaries available yet.")

st.write("---")
st.caption(
    "Demo built for executive review: secure PII handling, structured extraction, and a transparent processing pipeline."
)
