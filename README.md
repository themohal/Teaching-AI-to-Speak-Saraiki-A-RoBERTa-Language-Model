# Saraiki Next-Word Prediction

A Streamlit app for next-word / fill-in-the-blank prediction in the **Saraiki**
language, powered by a fine-tuned RoBERTa model:
[`themohal/saraiki-roberta-base-small-finetuned3`](https://huggingface.co/themohal/saraiki-roberta-base-small-finetuned3).

RoBERTa is a masked language model. To predict the *next word*, the app appends
the model's mask token to your text. You can also place the mask token yourself
to fill a blank anywhere in a sentence.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

The first run downloads the model from Hugging Face (cached afterward), then
opens the app in your browser at http://localhost:8501.

## Usage

- **Next word**: type Saraiki text (e.g. `میں سرائیکی`) and click **Predict**.
- **Fill a blank**: insert the mask token (shown in the app, typically `<mask>`)
  anywhere in your text, then click **Predict**.
- Adjust the number of predictions with the slider in the sidebar.
