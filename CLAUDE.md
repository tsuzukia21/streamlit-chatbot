# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
streamlit run st_Chatbot.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture Overview

This is a Streamlit-based chatbot application that recreates the ChatGPT browser experience with multiple AI model support.

**Process Flow Diagram**: See @flowchart.md for detailed Mermaid flowchart showing the complete application flow including user interactions, AI responses, message editing, and session management.

### Core Components

**Single-File Architecture**: The entire application logic is contained in `st_Chatbot.py`, implementing a stateful chat interface using Streamlit's session state management.

**Multi-Model Support**: Integrates with three AI providers through LangChain:
- OpenAI (gpt-4.1, gpt-4.5-preview)
- Anthropic Claude (claude-sonnet-4-20250514) - default model
- Google Gemini (gemini-2.5-pro-preview-05-06)

**Session State Management**: Critical state variables managed in Streamlit session:
- `chat_history`: List of tuples containing conversation messages
- `edit_states`: Dictionary tracking which messages are being edited
- `total_tokens`: Token usage counter with 50,000 limit
- `system_prompt`: Configurable system message
- `temperature`: Model temperature setting
- `llm`: Current language model instance

### Key Features Implementation

**Message Editing**: Users can edit any message in conversation history via `render_human_message()` function. When edited, all subsequent messages are deleted using `modify_message()`.

**Streaming Responses**: Uses LangChain's streaming interface with real-time token display and stop functionality.

**Token Management**: Implements token counting using tiktoken for GPT-4o encoding, with automatic conversation clearing when limits are exceeded.

**API Key Management**: Supports environment variables and UI input for all three providers' API keys.

### Critical Functions

- `show_chat_history()`: Main rendering function for conversation display
- `render_human_message()` / `render_assistant_message()`: Message-specific rendering with edit capabilities
- `update_model()`: Switches between AI providers and validates API keys
- `check_token()`: Enforces token and message limits
- `clear_chat()`: Resets conversation state

### Styling and UI

Custom CSS targeting Streamlit components for button positioning and layout. Uses Material Design icons for chat avatars and UI elements.

The application uses a wide layout configuration and implements custom markdown rendering to handle special characters in chat messages.