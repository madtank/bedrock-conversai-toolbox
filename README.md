# ToolboxAI

ToolboxAI is a Streamlit application that leverages the [AWS Bedrock Converse Stream API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html) with Claude 3, integrating powerful tool use capabilities for web search, web scraping, and image analysis. This versatile AI assistant can process and analyze uploaded images, providing insights and descriptions alongside its text-based interactions.

<img src="assets/icon.png" alt="ToolboxAI Logo" width="200" height="200">

## Key Features:

- Interactive chat interface powered by Claude 3 via AWS Bedrock
- Seamless integration of web search and web scraping tools
- Image upload and analysis capabilities
- Real-time conversation streaming
- Token usage tracking for efficient management

With ToolboxAI, users can engage in rich, multimodal conversations that combine text-based queries, web information retrieval, and visual content analysis.

## Tool Use Demonstration

See ToolboxAI in action with this image from our demonstration:

<img src="assets/bedrock_tool_use.png" alt="Bedrock Tool Use" width="600"/>

## New Feature: Image Upload and Analysis

ToolboxAI now supports image uploads, allowing for more diverse and rich interactions with the AI:

- **Seamless Image Integration**: Users can now upload images directly into the chat interface.
- **AI Image Analysis**: The AI can analyze and comment on uploaded images, providing insights and descriptions.
- **Flexible Format Support**: Supports common image formats including PNG, JPG, JPEG, and WebP.

<img src="assets/image_upload.png" alt="Image Upload Feature" width="600"/>

## Sending Messages with Documents

ToolboxAI has been enhanced to support sending messages with documents, broadening the scope of interactions with the AI. This feature allows users to include documents in their conversations, which the AI can then process and respond to accordingly.

### Supported Document Formats

The application supports a variety of document formats, including:
- PDF (`pdf`)
- CSV (`csv`)
- Microsoft Word (`doc`, `docx`)
- Microsoft Excel (`xls`, `xlsx`)
- HTML (`html`)
- Plain Text (`txt`)
- Markdown (`md`)

### How to Use

To send a document in a message:
1. Click on the "Upload a document" button in the chat interface.
2. Select a document file from your device in one of the supported formats.
3. The document will be sent as part of your message to the AI, which can then provide feedback or information based on the document's content.

This feature opens up new possibilities for document-based queries and analyses, enhancing the utility and versatility of ToolboxAI.

## Prerequisites

- Python 3.9+
- AWS account with access to Bedrock
- Configured AWS credentials

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/madtank/ToolboxAI.git
   cd ToolboxAI
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure your AWS credentials are properly configured.

## Running the Application

Launch the application with:

```
streamlit run main.py
```

This command starts the Streamlit server and opens the application in your default web browser.

To use this feature:
1. Click on the "Upload an image" button at the top of the chat interface.
2. Select an image file from your device.
3. The image will be displayed in the chat and can be referenced in your conversations with the AI.

This addition opens up new possibilities for visual-based queries and analyses. Try it out and explore how it enhances your interactions with ToolboxAI!

## Application Structure

ToolboxAI consists of five main components:

1. `main.py`: Streamlit UI and main application loop
2. `src/bedrock_client.py`: AWS Bedrock service interactions
3. `src/conversation_handler.py`: AI response processing and conversation flow management
4. `src/utils.py`: Utility functions for various application needs
5. `src/tools.py`: Tool-related functions and configurations for web search and scraping

## Customization

Extend ToolboxAI's capabilities by modifying `src/tools.py`. Remember to update the `toolConfig` dictionary when adding new tools.

## Troubleshooting

If you encounter issues:

1. Verify your AWS credentials are correctly set up
2. Ensure all dependencies are properly installed
3. Consult the Streamlit and AWS Bedrock documentation for service-specific troubleshooting

## Contributing

We welcome contributions to enhance ToolboxAI. Please follow the standard fork-and-pull request workflow for your submissions.

## License

ToolboxAI is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for full details.
