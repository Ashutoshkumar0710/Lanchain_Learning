import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1
)

st.title("Research Paper Summarizer")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)


style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-friendly",
        "Technical",
        "Code-oriented",
        "Mathematical"
    ]
)


length_input = st.selectbox(
    "Select explanation length",
    [
        "Short (1-2 Paragraphs)",
        "Medium (3-5 Paragraphs)",
        "Long (detailed explanation)"
    ]
)


# Prompt Template
template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}"
with the following specifications:

Explanation style: {style_input}
Explanation length: {length_input}

1. Mathematical Details:
- Include relevant mathematical equations if present in the paper.
- Explain the mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
- Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with:
"Insufficient information available" instead of guessing.

Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
    input_variables=[
        "paper_input",
        "style_input",
        "length_input"
    ]
)


# Summarize button
if st.button("Summarize"):

    prompt = template.invoke(
        {
            "paper_input": paper_input,
            "style_input": style_input,
            "length_input": length_input
        }
    )

    result = model.invoke(prompt)

    st.subheader("Summary")
    st.write(result.content)