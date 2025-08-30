__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
print("SQLite version:", sqlite3.sqlite_version)

import streamlit as st
from openai import OpenAI
from prompts import qa_testcase_prompt, qa_prompt, qa_strategy_prompt
from models import (
    active_model,process_document, process_url, get_vector_collection, add_to_vector_collection,
    query_collection, call_llm, re_rank_cross_encoders, list_vector_sources
)
from utils import generate_pdf, get_base64_image

import uuid



if __name__ == "__main__":

    # ------------------------
    # Init session state
    # ------------------------
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())  # Or use a login username

    user_id = st.session_state["user_id"]
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": qa_prompt}
        ]
    if "vector_index" not in st.session_state:
        st.session_state["vector_index"] = None
        st.session_state["chunks"] = []
    
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
        st.header("💾 Configuration Settings")


        with st.expander("💽 Add Resource"):
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
                add_to_vector_collection(all_splits, normalize_uploaded_file_name, user_id)

        if url_input and process_url_btn:
            with st.spinner("Fetching and analyzing web page..."):
                all_splits = process_url(url_input)
                add_to_vector_collection(all_splits, url_input.replace("https://", "").replace("http://", "").replace("/", "_"),user_id)

        with st.expander("🧑‍🦱 Model Selection (Offline/Online)"):
            mode = st.radio("Choose model source:", ["Offline", "Online (OpenAI)"])
            
            usr_key = None
            if mode == "Online (OpenAI)":
                usr_key = st.text_input("Enter your OpenAI API Key", type="password")
            if mode == "Online (OpenAI)" and usr_key:
                client = OpenAI(api_key=usr_key)
            else:
                client = None
            st.info("Offline model: "+active_model)
        with st.expander("📚 My Library Files"):
            sources = list_vector_sources(user_id)
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
            text-align: left;
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
        st.title("🧮 Quality Engineering AI Assistant")
        st.header("Your on-demand AI assistant for Quality Engineering")

    center,right= st.columns([6,2])

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

        prompt = st.text_area(
            "**Provide your context or questions on the source provided**",
            value=st.session_state.get("prompt_text", "")
        )

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
                results = query_collection(prompt,user_id)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context,prompt=prompt)
                response_text = "".join([chunk for chunk in call_llm(context=relevant_text,sysprompt=qa_prompt, prompt=prompt, spl_prompt="Brief the related context with examples or usecases if any.", mode=mode, client=client)])
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
                results = query_collection(prompt,user_id)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context,prompt=prompt)
                response = call_llm(context=relevant_text,sysprompt=qa_prompt,prompt=prompt, spl_prompt="Provide a detailed summary on the given requirement.", mode=mode, client=client)
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
                results = query_collection(prompt,user_id)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context,prompt=prompt)
                spl_prompt="Provide a detailed test strategy. Also mention the risks, assumptions and estimated efforts required to complete the testing."
                response = call_llm(context=relevant_text,sysprompt=qa_strategy_prompt, prompt=prompt, spl_prompt=spl_prompt, mode=mode, client=client)
                st.write_stream(response)

                with st.expander("Analyzed documents references:"):
                    st.write(results)

                with st.expander("Considered document references:"):
                    st.write(relevant_text_ids)
                    st.write(relevant_text)

        if test_case and prompt:
            with st.spinner("Test cases are crucial for ensuring quality... Let me assist you with that..."):
                results = query_collection(prompt,user_id)
                context = results.get("documents")[0]
                relevant_text, relevant_text_ids = re_rank_cross_encoders(context,prompt=prompt)
                #spl_prompt="List all possible usecases in the markdown table format containing the fields 'S.no, Summary, Description, Preconditions, Step Summary, Expected Results'. Each step summary should be of separate row with the expected result.  Make sure to cover positive, negative, boundary and edge cases sorted in same order. Also list down a section covering assumptions and risks if any."
                # spl_prompt={
                #         "task": "Generate use case documentation",
                #         "output_format": "markdown",
                #         "requirements": {
                #             "table_fields": [
                #             "S.no",
                #             "Summary",
                #             "Description",
                #             "Preconditions",
                #             "Step Summary",
                #             "Expected Results"
                #             ],
                #             "case_types": [
                #             "positive",
                #             "negative",
                #             "boundary",
                #             "edge"
                #             ],
                #             "sorting_order": "positive, negative, boundary, edge",
                #             "additional_sections": [
                #             {
                #                 "title": "Assumptions and Risks",
                #                 "content": "List any assumptions made during use case creation and potential risks associated with each case type."
                #             }
                #             ]
                #         },
                #         "format_style_guidelines": {
                #             "markdown_table": "true",
                #             "clear_and_concise": "true",
                #             "technical_depth": "medium",
                #             "audience": "QA engineers, developers, product managers"
                #         }
                #         }
                response = call_llm(context=relevant_text,sysprompt=qa_prompt, prompt=prompt, spl_prompt=qa_testcase_prompt, mode=mode, client=client)

                st.markdown("""
                        <style>
                        .stText {
                            text-align: left;
                        }
                        .stMarkdown{    
                            text-align: left;
                            color: black;
                        }
                        .stMarkdown p{    
                            text-align: left;
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

                st.write_stream(response)
                # response_text = ""
                # for chunk in response:
                #     response_text += chunk
                # #st.write(response_text)
                # st.code(response_text, language="markdown")
                # copy_code = f"""
                #     <button onclick="navigator.clipboard.writeText(`{response_text}`)">Copy to Clipboard</button>
                # """
                # st.markdown(copy_code, unsafe_allow_html=True)
                #  # Download as Excel
                # import pandas as pd
                # import io
                # df = pd.DataFrame({"Result": [response_text]})
                # excel_buffer = io.BytesIO()
                # df.to_excel(excel_buffer, index=False)
                # excel_buffer.seek(0)
                # st.download_button(
                #     "Download Result as Excel",
                #     data=excel_buffer,
                #     file_name="result.xlsx",
                #     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                # )


                st.markdown(
                    """
                    <br><br><br>
                    """
                    , unsafe_allow_html=True)
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
        with st.expander("📇 Your Prompts History"):
            st.markdown(
            """
            Your past prompts will be stored here for easy access. Click on any prompt to reuse it.
            """
            , unsafe_allow_html=True)
            # ------------------------
            # Dropdown of past prompts
            # ------------------------
            if st.session_state["prompts"]:
                selected_prompt = st.selectbox("📜 Past Prompts", st.session_state["prompts"][::-1])  # newest first
                st.info(f"Selected: {selected_prompt}")
                if st.button("Use Selected Prompt"):
                    st.session_state["prompt_text"] = selected_prompt
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