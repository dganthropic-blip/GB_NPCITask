"""CLI chat interface for the UPI Star Schema conversational agent."""
from build_schemas import load_or_build_schemas
from src.agent import UPIAgent


def main():
    print("Loading UPI star schemas...")
    schemas = load_or_build_schemas()
    agent = UPIAgent(schemas)
    print("Ready. Ask a question about UPI volume, value, ATS, or mandates.")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input or user_input.lower() in ("exit", "quit"):
            break
        result = agent.chat(user_input)
        print(f"\nAssistant: {result['response']}")
        if result["tools_used"]:
            print(f"\n  [Tools used: {len(result['tools_used'])} calls]")
            for t in result["tools_used"]:
                print(f"    → {t['tool']}({t['args']})")


if __name__ == "__main__":
    main()
