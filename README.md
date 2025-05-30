# Website Insights Chatbot

This Streamlit application allows users to interact with the content of a website by asking questions and receiving answers based on the website's content. The application uses LangChain, Google's Gemini model, and Chroma for vector storage and retrieval.

## Features

- **Web Content Loading**: Loads content from a given website URL.
- **Text Splitting**: Splits the content into smaller chunks for efficient processing.
- **Vector Storage**: Uses Chroma to store and retrieve document embeddings.
- **Chat Interface**: Provides a chat interface where users can ask questions related to the website's content.
- **Contextual Responses**: Ensures that the chatbot only provides answers relevant to the website's content.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.8 or higher
- Streamlit
- LangChain
- HuggingFace Transformers
- Chroma
- Google Generative AI

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/website-insights-chatbot.git
   cd website-insights-chatbot
   ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up your environment variables:
   Create a .env file in the root directory.
   Add your Google API key to the .env file:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```