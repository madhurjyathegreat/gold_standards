import os
import fitz  # PyMuPDF
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from fpdf import FPDF
import html

# === Constants and Config ===
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.1-8b-instant"

# === Session State Setup ===
if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(
        model_name=MODEL_NAME,
        temperature=0.3,
        api_key=GROQ_API_KEY
    )

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = ""

if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = {}

# === Helper Classes ===
class PDFExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_text(self):
        text = ""
        with fitz.open(self.file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

class SectionSplitter:
    def split_by_sections(self, raw_text):
        sections = {}
        current_section = None
        for line in raw_text.splitlines():
            if line.strip().startswith(tuple(str(i) for i in range(1, 10))):
                current_section = line.strip()
                sections[current_section] = ""
            elif current_section:
                sections[current_section] += line.strip() + " "
        return sections

# === PDF Writer Function ===
def generate_pdf(content_dict):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "HR Policy Comparison Report", ln=True, align="C")
    pdf.ln(10)

    for section_title, section_text in content_dict.items():
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, section_title, ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", size=12)
        sanitized = html.unescape(section_text)
        ascii_text = sanitized.encode("latin-1", errors="ignore").decode("latin-1")
        lines = ascii_text.split("\n")
        for line in lines:
            clean_line = line.strip("\n ")
            if clean_line:
                pdf.multi_cell(0, 10, clean_line)
        pdf.ln(5)

    output_path = "HR_Policy_Comparison_Report.pdf"
    pdf.output(output_path)
    return output_path

# === Cached Agent Functions ===
@st.cache_data(show_spinner=False)
def cached_delta(nbs_text, vm_text):
    prompt = PromptTemplate.from_template("""
    Compare the following HR policy sections from two banks. Highlight key differences and their potential implications:

    ---
    NBS:
    {nbs_text}

    VM:
    {vm_text}

    Output the differences in bullet points with rationale.
    """).format(nbs_text=nbs_text, vm_text=vm_text)
    return st.session_state.llm.predict(prompt)

@st.cache_data(show_spinner=False)
def cached_reasoning(delta_text):
    prompt = PromptTemplate.from_template("""
    Given the policy delta below, provide business, regulatory, or operational reasoning behind the change:

    {delta_text}

    Provide a thoughtful explanation in 2-3 bullet points.
    """).format(delta_text=delta_text)
    return st.session_state.llm.predict(prompt)

@st.cache_data(show_spinner=False)
def cached_summary(policy_text):
    prompt = PromptTemplate.from_template("""
    Summarize the following HR policy section into 3 concise bullet points:

    {policy_text}
    """).format(policy_text=policy_text)
    return st.session_state.llm.predict(prompt)

# === Streamlit UI ===
st.set_page_config(page_title="Agentic HR Policy Comparator", layout="wide")
st.title("🤖 HR Policy Comparison using Agentic AI")

# === Sidebar with PMI Policy Sections ===
st.sidebar.title("📂 Post-Merger Policy Categories")
selected_policy = st.sidebar.radio("Choose a policy domain:", (
    "HR & People Policies",
    "IT & Data Governance",
    "Finance & Procurement",
    "Risk & Compliance",
    "Customer Operations",
    "Corporate Governance"
))

if selected_policy != "HR & People Policies":
    st.warning(f"🔧 The '{selected_policy}' module is under development.")
    st.stop()

# === HR & People Policy Section ===
nbs_file = st.file_uploader("Upload NBS Bank HR Policy PDF", type="pdf")
vm_file = st.file_uploader("Upload VM Bank HR Policy PDF", type="pdf")

if nbs_file and vm_file:
    with open("nbs.pdf", "wb") as f:
        f.write(nbs_file.read())
    with open("vm.pdf", "wb") as f:
        f.write(vm_file.read())

    nbs_text = PDFExtractor("nbs.pdf").extract_text()
    vm_text = PDFExtractor("vm.pdf").extract_text()

    nbs_sections = SectionSplitter().split_by_sections(nbs_text)
    vm_sections = SectionSplitter().split_by_sections(vm_text)

    st.subheader("Section-wise Comparison")
    for section in nbs_sections:
        if section in vm_sections:
            with st.expander(f"📌 {section}"):
                delta = cached_delta(nbs_sections[section], vm_sections[section])
                reasoning = cached_reasoning(delta)
                summary_nbs = cached_summary(nbs_sections[section])
                summary_vm = cached_summary(vm_sections[section])

                section_output = f"""
                🔹 **{section}**

                **Delta:**
                {delta}

                **Reasoning:**
                {reasoning}

                **NBS Summary:**
                {summary_nbs}

                **VM Summary:**
                {summary_vm}
                """
                st.markdown(section_output)
                st.session_state.knowledge_base += section_output + "\n"
                # Store content for PDF generation
                if "pdf_content" not in st.session_state:
                    st.session_state.pdf_content = {}
                st.session_state.pdf_content[section] = section_output

    if st.button("📥 Download Comparison Report as PDF"):
        pdf_path = generate_pdf(st.session_state.pdf_content)
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf_path, mime="application/pdf")

# === Chatbot Section ===
st.subheader("💬 Ask Your HR Assistant")
user_query = st.chat_input("Ask about HR differences, summaries, or insights...")

if user_query:
    prompt = PromptTemplate.from_template("""
    Based on the following knowledge base of HR policy differences, answer the user's question clearly:

    ---
    {kb}

    Question: {question}

    Answer in a helpful and concise way:
    """)
    max_chars = 5000
    kb_snippet = st.session_state.knowledge_base[-max_chars:]

    chain = LLMChain(llm=st.session_state.llm, prompt=prompt)
    response = chain.run({"kb": kb_snippet, "question": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)
    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.memory.chat_memory.add_user_message(user_query)
    st.session_state.memory.chat_memory.add_ai_message(response)

with st.expander("🧠 Chat Memory"):
    for msg in st.session_state.memory.chat_memory.messages:
        st.markdown(f"**{msg.type.upper()}**: {msg.content}")
