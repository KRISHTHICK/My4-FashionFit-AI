# 👗 FashionFit AI

A stylish Streamlit app that extracts outfit info from images and uses TinyLLaMA (via Ollama) for fashion Q&A, style tips, and social media post generation. No cloud APIs required.

## Features

- Upload outfit images
- Ask fashion-related questions (local LLM)
- Auto-generate Instagram captions
- Visualize key fashion terms (colors, brands, etc.)

## Run Locally

```bash
pip install -r requirements.txt
ollama run tinyllama
streamlit run app.py
