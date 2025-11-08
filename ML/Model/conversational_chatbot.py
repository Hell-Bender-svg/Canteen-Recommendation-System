import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def run_chatbot():
    try:
        client = genai.Client()
        MODEL_NAME = "gemini-2.5-flash"
        print(f"🤖 Initializing Gemini Chatbot with model: {MODEL_NAME}...")
    except Exception:
        print("❌ ERROR: Failed to initialize Gemini Client.")
        print("Please ensure you have set your GEMINI_API_KEY in your .env file.")
        sys.exit(1)

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

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("\nChip: Goodbye! Have a great day!")
                break

            if not user_input:
                continue

            try:
                response_stream = chat.send_message_stream(user_input)
                print("Chip: ", end="")
                full_response = ""

                for chunk in response_stream:
                    if chunk.text:
                        print(chunk.text, end="")
                        full_response += chunk.text

                sys.stdout.flush()
                print()

            except Exception as stream_error:
                print(f"\nChip: I ran into an API error while responding: {stream_error}. Please try again.")
                continue

        except KeyboardInterrupt:
            print("\nChip: Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nChip: Oops! An unexpected error occurred: {e}. Restarting loop.")

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("CRITICAL: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    run_chatbot()
