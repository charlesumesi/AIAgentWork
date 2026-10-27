import os
import uuid
import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

st.set_page_config(
    page_title="CU AI Agent",
    layout="centered",
)

st.title("CU AI Agent")
st.caption("Chat with your deployed AI agent")

# AWS configuration
AWS_REGION = st.secrets["AWS_REGION"]
DEFAULT_HARNESS_ARN = st.secrets["AGENTCORE_HARNESS_ARN"]

# Create a reusable AWS client
@st.cache_resource
def get_bedrock_client(region):
    return boto3.client(
        "bedrock-agentcore",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=region,
    )

client = get_bedrock_client(AWS_REGION)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "runtime_session_id" not in st.session_state:
    st.session_state.runtime_session_id = str(uuid.uuid4())

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")

    harness_arn = st.text_input(
        "Harness ARN",
        value=DEFAULT_HARNESS_ARN,
        help="Paste the ARN of your deployed AgentCore Harness.",
    )

    harness_arn = DEFAULT_HARNESS_ARN

    st.caption(f"AWS Region: {AWS_REGION}")

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.runtime_session_id = str(uuid.uuid4())
        st.rerun()

    st.caption("A new conversation starts a new harness session.")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Customer input
prompt = st.chat_input("Ask your AI assistant...")

if prompt:
    # Display and save the customer's message
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the harness and display its streamed response
    with st.chat_message("assistant"):
        output = st.empty()
        #output.markdown = ("\U0001F4AD" + " Thinking...")
        answer = ""

        try:
            response = client.invoke_harness(
                harnessArn=harness_arn.strip(),
                runtimeSessionId=st.session_state.runtime_session_id,
                model={
                    "bedrockModelConfig": {
                        "modelId": "amazon.nova-pro-v1:0"
                        }
                    },
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ],
            )

            for event in response["stream"]:
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"].get(
                        "delta", {}
                    )

                    if "text" in delta:
                        answer += delta["text"]
                        output.markdown(answer + " \u258C")

                elif "runtimeClientError" in event:
                    error_message = event[
                        "runtimeClientError"
                    ].get("message", "Unknown runtime error")
                    raise RuntimeError(error_message)

            if answer:
                output.markdown(answer)
            else:
                answer = (
                    "The harness returned no text. "
                    "Check the harness logs and response events."
                )
                output.warning(answer)

        except (ClientError, BotoCoreError, RuntimeError) as exc:
            answer = f"Error communicating with the harness: {exc}"
            output.error(answer)

        except Exception as exc:
            answer = f"Unexpected error: {exc}"
            output.error(answer)

    # Save the assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
