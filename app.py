import streamlit as st
from openai import OpenAI
from prompts import system_prompt, qa_prompt, qa_strategy_prompt
from models import (
    process_document, process_url, get_vector_collection, add_to_vector_collection,
    query_collection, call_llm_strategy, call_llm, re_rank_cross_encoders, list_vector_sources
)
from utils import generate_pdf, get_base64_image


if __name__ == "__main__":

    # ------------------------
    # Init session state
    # ------------------------
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": qa_prompt}
        ]
    if "vector_index" not in st.session_state:
        st.session_state["vector_index"] = None
        st.session_state["chunks"] = []
    # st.markdown("""
    #     <style>
    #     .stSidebar  {
    #         text-align: center;
    #         color: white;
    #     }
    #     .stFileUploader  {
    #         text-align: center;
    #         color: white;
    #     }
    #     .stFileUploader  p{
    #         color: white;
    #     }
    #     .stMarkdown{    
    #         text-align: left;
    #         color: white;
    #     }
    #      .stButton button {
    #         background-color: #d3d3d3;
    #         color: black;
    #         border-radius: 10px;
    #         border: none;
    #         padding: 0.5em 1em;
    #         font-size: 1em;
    #         margin: 0.2em;
    #         width: 200px;
    #         height: 90px;
    #         transition: background 0.5s;
    #     }
    #     .stButton button:hover {
    #         background-color: #a3a3a3;
    #         color: white;
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)
    
   # Document Upload Area
    st.set_page_config(
            page_title="☄️ Quality Engineering AI Assistant by Harry ☄️", 
            page_icon="☄️",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    with st.sidebar:
        st.set_page_config(page_title="QE AI Assistant", layout="wide")
        st.markdown(
        """<h2><br><br><br></h2>"""
        , unsafe_allow_html=True)
        st.header("Configuration Settings for your Quality Engineering AI Assistant")


        with st.expander("Add Resource"):
            with st.expander("Add your PDF/Excel/Text Resource:"):
                uploaded_file = st.file_uploader(
                    "Add your source to help you better", 
                    type=["pdf", "xlsx", "xls", "txt"], 
                    accept_multiple_files=False
                )
                process = st.button(
                    "➕ Add Document Content",
                )

            with st.expander("Add your Webpage Resource:"):
                url_input = st.text_input("OR Enter a web page URL to analyze")
                process_url_btn = st.button("➕ Add Web Content")
    
        
        if uploaded_file and process:
            normalize_uploaded_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            with st.spinner("Analyzing your document... Please allow me sometime..."):
                all_splits = process_document(uploaded_file)
                add_to_vector_collection(all_splits, normalize_uploaded_file_name)

        if url_input and process_url_btn:
            with st.spinner("Fetching and analyzing web page..."):
                all_splits = process_url(url_input)
                add_to_vector_collection(all_splits, url_input.replace("https://", "").replace("http://", "").replace("/", "_"))

        with st.expander("Model Selection"):
            mode = st.radio("Choose model source:", ["Offline", "Online (OpenAI)"])
            usr_key = None
            if mode == "Online (OpenAI)":
                usr_key = st.text_input("Enter your OpenAI API Key", type="password")
            if mode == "Online (OpenAI)" and usr_key:
                client = OpenAI(api_key=usr_key)
            else:
                client = None
    
        with st.expander("📚 My Library Files"):
            sources = list_vector_sources()
            if sources:
                for src in sources:
                    st.markdown(f"- {src}")
            else:
                st.info("No sources processed yet.")
    # bg_image_base64 = get_base64_image("bg.gif")  # Make sure bg.gif is in your project folder

    st.markdown("""
        <style>
        .stApp {
            text-align: center;
            color: black;
            # background-image: url('https://gifdb.com/images/high/office-desk-background-myhb5mf4xpj1ews9.gif');
            # background-size: cover;
            # background-repeat: no-repeat;
            # background-attachment: fixed;
        }
        .stTextArea textarea {
            border-radius: 10px;
            border: 1px solid #90caf9;
            background: white;
            font-size: 1em;
            color: black;
            padding: 1em;
            height: 150px;
            width: 100%;
            transition: border-color 0.5s;
        }
        .stButton button {
            background-color: black;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 1em 1em;
            font-size: 1em;
            margin: 0.2em;
            width: 220px;
            height: 90px;
            transition: background 0.5s;
        }
        .stButton button:hover {
            background-color: #6acfeeff;
            color: white;
        }
        .stTitle, .stHeader, .stSubheader, .stText {
            text-align: center;
        }
        .stAlert {
            text-align: center;
            color: red;
        }
        .stMarkdown{    
            text-align: left;
            color: black;
        }
        .stMarkdown p{    
            text-align: center;
            color: black;
        }
        .stMarkdown hr{    
            border-color: black;
            border-block-end-width: 2px;
        }
        label {
            color: black;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
   
    topl,topc,topr= st.columns([1,6,1])
    with topc:
        st.title("☄️ Quality Engineering AI Assistant ☄️")
        st.header("Your on-demand AI assistant for Quality Engineering")

    center,right= st.columns([6,2])


    # with left:
    #     st.markdown(
    #     """<h2><br><br><br>Configuration Settings for your Quality Engineering AI Assistant</h2>"""
    #     , unsafe_allow_html=True)

    #     with st.expander("Add your PDF Resource:"):
    #         uploaded_file = st.file_uploader(
    #             "Add your source to help you better", type=["pdf"], accept_multiple_files=False
    #         )
    #         process = st.button(
    #             "➕ Add PDF Content",
    #         )

    #     with st.expander("Add your Webpage Resource:"):
    #         url_input = st.text_input("OR Enter a web page URL to analyze")
    #         process_url_btn = st.button("➕ Add Web Content")
    
        
    #     if uploaded_file and process:
    #         normalize_uploaded_file_name = uploaded_file.name.translate(
    #             str.maketrans({"-": "_", ".": "_", " ": "_"})
    #         )
    #         with st.spinner("Analyzing your document... Please allow me sometime..."):
    #             all_splits = process_document(uploaded_file)
    #             add_to_vector_collection(all_splits, normalize_uploaded_file_name)

    #     if url_input and process_url_btn:
    #         with st.spinner("Fetching and analyzing web page..."):
    #             all_splits = process_url(url_input)
    #             add_to_vector_collection(all_splits, url_input.replace("https://", "").replace("http://", "").replace("/", "_"))

    #     mode = st.radio("Choose model source:", ["Offline (Ollama)", "Online (OpenAI)"])


    with center:
        

        # ------------------------
        # Session state
        # ------------------------
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "system", "content": qa_prompt}
            ]
        if "prompts" not in st.session_state:
            st.session_state["prompts"] = []  # store all past prompts

        prompt = st.text_area("**Provide your context or questions on the source provided**")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ask = st.button("💡 Enlight me")
        with col2:
            summary = st.button("⏱  Quick Summary")
        with col3:
            test_strategy = st.button("💻  Generate Test Strategy")
        with col4:
            test_case = st.button("📤  Generate Test Cases")

        if not(uploaded_file or url_input) and not prompt and (ask or  summary or  test_strategy or  test_case):
            st.error(
                "Please upload a document and provide a question or context to help you better"
            )
    resultl,resultr= st.columns([6,1])
    with resultl:
        if ask and prompt:
            with st.spinner("ah okay, let me think..."):
                results = query_collection(prompt)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
                response_text = "".join([chunk for chunk in call_llm(context=relevant_text, prompt=prompt)])
                st.write(response_text)

                pdf_bytes = generate_pdf(response_text)
                st.download_button("Download Result as PDF", data=pdf_bytes, file_name="result.pdf", mime="application/pdf")

                with st.expander("Analyzed documents references:"):
                    st.write(results)

                with st.expander("Considered document references:"):
                    st.write(relevant_text_ids)
                    st.write(relevant_text)
        
        if summary and prompt:
            with st.spinner("Sure, I'm on it... Curating a summary for you..."):
                results = query_collection(prompt)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
                response = call_llm(context=relevant_text, prompt="Provide a detailed summary on the given requirement."+prompt)
                st.write_stream(response)

            # pdf_bytes = generate_pdf(response)
            # st.download_button("Download Result as PDF", data=pdf_bytes, file_name="AI-Summary.pdf", mime="application/pdf")

                with st.expander("Analyzed documents references:"):
                    st.write(results)

                with st.expander("Considered document references:"):
                    st.write(relevant_text_ids)
                    st.write(relevant_text)

        if test_strategy and prompt:
            with st.spinner("Test strategies are essential in the testing phase... Let me help you with that..."):
                results = query_collection(prompt)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
                spl_prompt="Provide a detailed test strategy. Also mention the risks, assumptions and estimated efforts required to complete the testing."
                response = call_llm_strategy(context=relevant_text, prompt=spl_prompt)
                st.write_stream(response)

                with st.expander("Analyzed documents references:"):
                    st.write(results)

                with st.expander("Considered document references:"):
                    st.write(relevant_text_ids)
                    st.write(relevant_text)

        if test_case and prompt:
            with st.spinner("Test cases are crucial for ensuring quality... Let me assist you with that..."):
                results = query_collection(prompt)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context)
                spl_prompt="List all possible usecases in the table format containing the fields 'S.no, Testcase Summary, Test Conditions, Business Usecase' along with assumptions if any."
                response = call_llm(context=relevant_text, prompt=spl_prompt)
                st.write_stream(response)

                with st.expander("Analyzed documents references:"):
                    st.write(results)

                with st.expander("Other Related document references:"):
                    st.write(relevant_text_ids)
                    st.write(relevant_text)

        
    with right:
        st.markdown(
        """
        <br>
        """
        , unsafe_allow_html=True)
        # ------------------------
        # Dropdown of past prompts
        # ------------------------
        if st.session_state["prompts"]:
            selected_prompt = st.selectbox("📜 Past Prompts", st.session_state["prompts"][::-1])  # newest first
            st.info(f"Selected: {selected_prompt}")

        # # ------------------------
        # # Display full conversation
        # # ------------------------
        # st.subheader("Conversation History")
        # for msg in st.session_state["messages"]:
        #     #role = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
        #     if msg["role"] == "user":
        #         st.markdown(f"** 🧑User:** {msg['content']}")

    st.markdown(
        """
        ---
        Made with ❤️ by [Harry](harry.viswa@gmail.com)
        """
    )