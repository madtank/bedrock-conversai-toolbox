import streamlit as st
from datetime import date
from src.bedrock_client import create_bedrock_client
from src.conversation_handler import handle_chat_input, process_ai_response, handle_document_message
from src.utils import new_chat, format_search_results
import base64
import random
from PIL import Image

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

icon = Image.open("assets/favicon.ico")
st.set_page_config(page_title="ToolboxAI", page_icon=icon)

def main():
    st.title("ToolboxAI")

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = random.randint(1, 100000)
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = None

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"], key=f"uploader_{st.session_state['uploader_key']}")

    # File uploader for documents
    uploaded_document = st.file_uploader("Upload a document", type=["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"], key=f"document_{st.session_state['uploader_key']}")

    # Sidebar elements
    st.sidebar.button("New Chat", type="primary", on_click=new_chat)

    # Model selection
    models = [
        {"name": "Claude 3 Haiku", "id": "anthropic.claude-3-haiku-20240307-v1:0"},
        {"name": "Claude 3 Opus", "id": "anthropic.claude-3-opus-20240229-v1:0"},
        {"name": "Claude 3 Sonnet", "id": "anthropic.claude-3-sonnet-20240229-v1:0"},
        {"name": "Claude 3.5 Sonnet", "id": "anthropic.claude-3-5-sonnet-20240620-v1:0"}
    ]
    model_names = [model["name"] for model in models]
    selected_model_name = st.sidebar.selectbox("Select a model", model_names)
    # Display token usage if available
    if st.session_state.token_usage:
        st.sidebar.markdown("### Token Usage")
        st.sidebar.markdown(f"Input Tokens: {st.session_state.token_usage['inputTokens']}")
        st.sidebar.markdown(f"Output Tokens: {st.session_state.token_usage['outputTokens']}")
        st.sidebar.markdown(f"Total Tokens: {st.session_state.token_usage['totalTokens']}")


    model_id = next((model["id"] for model in models if model["name"] == selected_model_name), None)
    region_name = "us-west-2"

    # System prompt and other configurations
    system_prompt = f"""
    Answer as many questions as you can using your existing knowledge.
    Only search the web for queries that you can not confidently answer.
    Today's date is {date.today().strftime("%B %d %Y")}
    If you think a user's question involves something in the future that hasn't happened yet, use the search tool.
    You can also analyze images that the user uploads.
    """
    
    bedrock_client = create_bedrock_client(region_name)
    system_prompts = [{"text": system_prompt}]
    inference_config = {"temperature": 0.1}
    additional_model_fields = {"top_k": 200}

    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "image" in message:
                st.image(message["image"], caption="Uploaded Image", width=300)
            if "tool_results" in message:
                with st.expander(f"🔍 Tool Results: {message['tool_name']}", expanded=False):
                    if message['tool_name'] == 'search':
                        st.markdown(format_search_results(message["tool_results"]))
                    else:
                        st.subheader("Input:")
                        st.code(message["tool_input"])
                        st.subheader("Results:")
                        st.json(message["tool_results"])

    # Chat input and processing
    if prompt := st.chat_input("Ask me anything!"):
        image_content = None
        image_format = None
        document_content = None
        document_format = None
        if uploaded_file is not None:
            image_content = uploaded_file.getvalue()
            image_format = uploaded_file.type.split('/')[-1]
        if uploaded_document is not None:
            document_content = uploaded_document.getvalue()
            document_format = uploaded_document.type.split('/')[-1]

        try:
            # Immediately display user message and image
            with st.chat_message("user"):
                st.markdown(prompt)
                if image_content:
                    image_b64 = base64.b64encode(image_content).decode()
                    st.image(f"data:image/{image_format};base64,{image_b64}", caption="Uploaded Image", width=300)
                if document_content:
                    # Handle document message
                    document_message = handle_document_message(document_content, document_format)
                    st.session_state.history.append(document_message)

            # Update session state
            handle_chat_input(prompt, image_content, image_format)

            # Process AI response
            process_ai_response(bedrock_client, model_id, st.session_state.history, system_prompts, inference_config, additional_model_fields)

            # Update the uploader key to reset the file uploader
            st.session_state['uploader_key'] = random.randint(1, 100000)
            st.rerun()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.exception("An error occurred during chat processing")

if __name__ == "__main__":
    main()
