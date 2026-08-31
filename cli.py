"""
Simple command-line interface for the FAQ chatbot.

Run with:
    python cli.py
"""

from chatbot import FAQChatbot


def main():
    print("=" * 50)
    print(" FAQ Chatbot (type 'quit' or 'exit' to stop) ")
    print("=" * 50)

    bot = FAQChatbot()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye! 👋")
            break

        result = bot.get_response(user_input)
        print(f"Bot: {result['answer']}")

        # Uncomment to debug matching scores:
        # print(f"    (matched: {result['matched_question']!r}, score: {result['score']:.2f})")


if __name__ == "__main__":
    main()
