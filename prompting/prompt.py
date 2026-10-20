import streamlit as st
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.1
)

st.title("Research Paper Summarizer")

user_input = st.text_area(
    "Paste your research paper text"
)

if st.button("Summarize"):
    if user_input:
        response = model.invoke(
            f"""
            Summarize the following research paper.

            Include:
            1. Problem statement
            2. Methodology
            3. Dataset
            4. Results
            5. Conclusion
            6. Limitations

            Research paper:
            {user_input}
            """
        )

        st.subheader("Summary")
        st.write(response.content)
    else:
        st.warning("Please enter the research paper.")