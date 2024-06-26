import boto3
from src.tools import toolConfig

def create_bedrock_client(region, handle_documents=False):
    """
    Creates a Bedrock client with optional document handling capabilities.
    
    :param region: AWS region where the Bedrock service is hosted.
    :param handle_documents: Flag to enable document handling features.
    :return: A Bedrock client instance.
    """
    return boto3.client(service_name='bedrock-runtime', region_name=region)

def get_stream(bedrock_client, model_id, messages, system_prompts, inference_config, additional_model_fields, include_documents=False):
    """
    Initiates a streaming conversation with the Bedrock service, optionally including document data in the message payload.
    
    :param bedrock_client: The Bedrock client instance.
    :param model_id: The model ID to use for the conversation.
    :param messages: The list of messages to send, possibly including document data.
    :param system_prompts: System prompts to provide context for the conversation.
    :param inference_config: Configuration for the inference request.
    :param additional_model_fields: Additional fields to include in the model request.
    :param include_documents: Flag to include document data in the message payload.
    :return: A generator for the streaming conversation events.
    """
    request_params = {
        "modelId": model_id,
        "messages": messages,
        "system": system_prompts,
        "inferenceConfig": inference_config,
        "additionalModelRequestFields": additional_model_fields,
        "toolConfig": toolConfig
    }
    if include_documents:
        # Modify the request to handle documents if required
        pass  # Placeholder for document handling logic

    response = bedrock_client.converse_stream(**request_params)
    return response.get('stream')

def stream_conversation(stream):
    if stream:
        for event in stream:
            yield event
