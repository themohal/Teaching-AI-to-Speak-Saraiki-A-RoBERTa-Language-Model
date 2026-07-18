"""
Streamlit app for next-word / masked-word prediction using a fine-tuned
Saraiki RoBERTa model.

RoBERTa is a masked language model, so "next word prediction" is done by
appending the tokenizer's mask token to the user's text (or letting the user
place the mask themselves) and asking the model to fill it.

Model: https://huggingface.co/themohal/saraiki-roberta-base-small-finetuned3
"""

import html

import streamlit as st

MODEL_ID = "themohal/saraiki-roberta-base-small-finetuned3"

st.set_page_config(
    page_title="Saraiki Next-Word Prediction",
    page_icon="✍️",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading the Saraiki RoBERTa model… (first run downloads it)")
def load_pipeline():
    """Load and cache the fill-mask pipeline once per session."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    return fill_mask, tokenizer.mask_token


def predict(text: str, mask_token: str, top_k: int):
    """Run fill-mask. If no mask is present, append one to predict the next word."""
    fill_mask, _ = st.session_state["_pipe"]
    if mask_token not in text:
        # Predict the next word: append a mask at the end.
        query = f"{text.rstrip()} {mask_token}"
    else:
        query = text

    results = fill_mask(query, top_k=top_k)

    # When multiple masks are present, transformers returns a list of lists.
    if results and isinstance(results[0], list):
        results = results[0]
    return query, results


# --- Sidebar --------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of predictions", min_value=1, max_value=20, value=5)
    st.markdown("---")
    st.markdown(
        f"**Model**\n\n[`{MODEL_ID}`](https://huggingface.co/{MODEL_ID})\n\n"
        "A RoBERTa masked-language model fine-tuned for the **Saraiki** language."
    )
    st.markdown(
        "**How it works**\n\n"
        "RoBERTa predicts a masked token. To predict the *next word*, "
        "the app appends a mask to your text. You can also place the mask "
        "yourself anywhere in the sentence to fill a blank."
    )


# --- Main -----------------------------------------------------------------
st.title("✍️ Saraiki Next-Word Prediction")
st.caption("Fine-tuned RoBERTa • fill-mask • Saraiki (سرائیکی)")

# Load the model (cached). Store pipe + mask token in session for reuse.
try:
    fill_mask, mask_token = load_pipeline()
    st.session_state["_pipe"] = (fill_mask, mask_token)
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to load the model: {e}")
    st.stop()

st.info(
    f"Type Saraiki text and I'll predict the next word. "
    f"To fill a blank instead, insert the mask token `{mask_token}` anywhere.",
    icon="💡",
)

default_text = "میں سرائیکی"
text = st.text_area(
    "Enter Saraiki text",
    value=default_text,
    height=120,
    help=f"Leave it as-is to predict the next word, or add `{mask_token}` to fill a blank.",
)

# Render the input right-to-left (Saraiki uses Perso-Arabic script).
if text.strip():
    st.markdown(
        f"<div dir='rtl' style='font-size:1.3rem; padding:0.5rem 0; "
        f"font-family:\"Noto Nastaliq Urdu\", serif;'>{html.escape(text)}</div>",
        unsafe_allow_html=True,
    )

col1, col2 = st.columns([1, 4])
with col1:
    go = st.button("🔮 Predict", type="primary", use_container_width=True)

if go:
    if not text.strip():
        st.warning("Please enter some Saraiki text first.")
    else:
        with st.spinner("Predicting…"):
            query, results = predict(text, mask_token, top_k)

        st.subheader("Predictions")
        st.caption(f"Query sent to model: `{query}`")

        # Raw scores are probabilities over the whole vocabulary, so they look
        # tiny. Normalize across the shown predictions to express relative
        # confidence among these options (sums to ~100%).
        total = sum(r["score"] for r in results) or 1.0

        for i, r in enumerate(results, start=1):
            token = r["token_str"].strip()
            score = r["score"]
            rel = score / total
            sequence = r.get("sequence", "").replace("<s>", "").replace("</s>", "").strip()

            st.markdown(
                f"<div dir='rtl' style='font-size:1.15rem; "
                f"font-family:\"Noto Nastaliq Urdu\", serif;'>"
                f"<b>{i}.</b> {html.escape(token)} "
                f"<span style='color:#888; font-size:0.85rem;'>({rel:.1%})</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(max(rel, 0.0), 1.0))
            with st.expander("Full sentence"):
                st.markdown(
                    f"<div dir='rtl' style='font-size:1.1rem;'>{html.escape(sequence)}</div>",
                    unsafe_allow_html=True,
                )

st.markdown("---")
st.caption(
    "Built with Streamlit + 🤗 Transformers. Model by "
    "[themohal](https://huggingface.co/themohal)."
)
