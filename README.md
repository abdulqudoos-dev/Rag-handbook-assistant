# RAG Handbook Assistant

> **Built by [Abdul Qudoos](https://www.abdul-qudoos.com)**, AI Automation & Forward Deployed Engineer · [Read the case study](https://www.abdul-qudoos.com/work/rag-handbook-assistant) · [More projects](https://www.abdul-qudoos.com/work)

Grounded question answering over a 62-page university Final Year Project handbook, built for a Generative AI course.

## How it works

1. **Ingest**: `ingest_py.py` extracts text from the PDF and splits it into 300-word chunks with 30% overlap (chunks under 50 words are dropped).
2. **Embed**: chunks are embedded with `all-MiniLM-L6-v2` (384 dimensions) and stored in a FAISS flat index (`faiss_index.bin`, under 1 MB).
3. **Retrieve**: each question retrieves the top 5 chunks by cosine similarity. If no chunk scores at least 0.25, the assistant says the handbook doesn't cover it instead of guessing.
4. **Answer**: GPT-4o-mini (temperature 0.3) answers from the retrieved text only, in a Streamlit UI (`app_py.py`).

Configuration and results are documented in `technical_report_pdf.md`, and every query is logged to `prompt_log.txt`.

## Run it

```bash
pip install -r requirements_txt.txt
# add OPENAI_API_KEY to a .env file
python ingest_py.py        # builds the FAISS index
streamlit run app_py.py
```

---

## About the author

I'm **Abdul Qudoos**, an AI automation and forward deployed engineer based in Islamabad, Pakistan. I build production AI agents, voice agents, workflow automation, and the full-stack products around them.

- Portfolio: [abdul-qudoos.com](https://www.abdul-qudoos.com)
- Case studies: [abdul-qudoos.com/work](https://www.abdul-qudoos.com/work)
- LinkedIn: [Abdul Qudoos](https://www.linkedin.com/in/abdul-qudoos-9a4640324/)
- Email: abdulqudoos7113@gmail.com
