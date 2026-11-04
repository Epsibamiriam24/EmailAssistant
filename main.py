from __future__ import annotations

from datetime import datetime
from pathlib import Path

from input_module import collect_customer_email
from intent_module import detect_intent, fallback_intent
from reply_module import fallback_reply, generate_reply
from retrieval_module import available_intents, find_similar_email, load_dataset
from summary_module import fallback_summary, summarize
from tone_module import TONES, fallback_tone, improve_tone
from utils import OllamaError


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Email_Assistant_Dataset_100.xlsx - emails.csv"
OUTPUT = ROOT / "outputs" / "generated_reply.txt"


def choose_tone() -> str:
    print("\nChoose reply tone:")
    for number, tone in enumerate(TONES, 1):
        print(f"{number}. {tone}")
    choice = input("Tone: ").strip()
    if choice not in {"1", "2", "3"}:
        raise ValueError("Please choose 1, 2, or 3.")
    return TONES[int(choice) - 1]


def save_result(customer: dict[str, str], intent: str, reply: str, summary: str) -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(
        f"Customer Name:\n{customer['name']}\n\n"
        f"Subject:\n{customer['subject']}\n\n"
        f"Customer Email:\n{customer['email']}\n\n"
        f"Intent:\n{intent}\n\n"
        f"Generated Reply:\n{reply}\n\n"
        f"Conversation Summary:\n{summary}\n\n"
        f"Date:\n{datetime.now().strftime('%d-%m-%Y')}\n",
        encoding="utf-8",
    )


def generate_email_reply(data) -> None:
    customer = collect_customer_email()
    intents = available_intents(data)
    try:
        intent = detect_intent(customer["email"], intents)
    except (OllamaError, ValueError) as exc:
        print(f"\nOllama intent detection unavailable ({exc}); using a local fallback.")
        intent = fallback_intent(customer["email"], intents)

    similar = find_similar_email(customer["email"], data)
    print(f"\nDetected Intent: {intent}")
    print(f"Similar Email ({similar['score']:.2f} similarity): {similar['email']}")
    print(f"Previous Reply: {similar['reply']}")

    try:
        reply = generate_reply(customer["email"], intent, str(similar["reply"]))
    except OllamaError as exc:
        print(f"Ollama reply generation unavailable ({exc}); using the retrieved reply.")
        reply = fallback_reply(intent, str(similar["reply"]))

    tone = choose_tone()
    try:
        reply = improve_tone(reply, tone)
    except OllamaError as exc:
        print(f"Ollama tone improvement unavailable ({exc}); using a local tone adjustment.")
        reply = fallback_tone(reply, tone)

    try:
        summary = summarize(customer["email"], reply)
    except OllamaError as exc:
        print(f"Ollama summary unavailable ({exc}); using a local summary.")
        summary = fallback_summary(customer["email"], reply)

    save_result(customer, intent, reply, summary)
    print("\nGenerated Reply:\n" + reply)
    print("\nConversation Summary:\n" + summary)
    print(f"\nSaved to: {OUTPUT}")


def main() -> None:
    try:
        data = load_dataset(DATASET)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Startup error: {exc}")
        return

    while True:
        print("\n========================================")
        print("INTELLIGENT EMAIL ASSISTANT")
        print("========================================")
        print("1. Generate Email Reply")
        print("2. View Last Conversation Summary")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            try:
                generate_email_reply(data)
            except (ValueError, EOFError) as exc:
                print(f"Input error: {exc}")
        elif choice == "2":
            if OUTPUT.exists():
                print("\n" + OUTPUT.read_text(encoding="utf-8"))
            else:
                print("\nNo generated conversation is available yet.")
        elif choice == "3":
            print("Goodbye.")
            return
        else:
            print("Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
