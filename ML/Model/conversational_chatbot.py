import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Configuration ---
load_dotenv() 

# --- Chatbot Core Functionality ---

def run_chatbot():
    # 1. Initialize Gemini Client
    try:
        client = genai.Client()
        MODEL_NAME = "gemini-2.5-flash" 
        print(f"🤖 Initializing Gemini Chatbot with model: {MODEL_NAME}...")
    except Exception as e:
        print("❌ ERROR: Failed to initialize Gemini Client.")
        print("Please ensure you have set your GEMINI_API_KEY in your .env file.")
        sys.exit(1)

    # 2. Create Chat Session (Correctly indented under run_chatbot)
    chat = client.chats.create(
        model=MODEL_NAME, 
        config=types.GenerateContentConfig(
            system_instruction="You are a friendly, witty, and helpful AI assistant named Chip. Keep your responses concise and engaging."
        )
    )

    print("\n------------------------------------------------------")
    print("👋 Hi! I'm Chip, your friendly AI assistant. Ask me anything!")
    print("   Type 'exit' or 'quit' to end the session.")
    print("------------------------------------------------------")

    # 3. Main Chat Loop
    while True:
        try: # Outer try block starts here
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nChip: Goodbye! Have a great day!")
                break
            
            if not user_input:
                continue

            # --- Improved Streaming and Error Handling ---
            
            try: # Inner try block for streaming starts here
                # Send message and get the response
                # Corrected streaming method
                response_stream = chat.send_message_stream(user_input)
                
                print("Chip: ", end="")
                
                # Print the response in real-time as chunks arrive
                full_response = ""
                for chunk in response_stream:
                    if chunk.text:
                        print(chunk.text, end="")
                        full_response += chunk.text
                
                sys.stdout.flush()
                print() 

            except Exception as stream_error: # Inner except block is aligned with inner try
                # Handle API errors that happen during streaming/generation
                print(f"\nChip: I ran into an API error while responding: {stream_error}. Please try your question again.")
                continue

        except KeyboardInterrupt: # Outer except block is aligned with outer try
            # Handle Ctrl+C gracefully
            print("\nChip: Session interrupted. Goodbye!")
            break
        except Exception as e: # Outer except block for unexpected errors
            print(f"\nChip: Oops! An unexpected error occurred: {e}. Restarting loop.")

# --- Execute Chatbot ---

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("CRITICAL: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
        
    run_chatbot()