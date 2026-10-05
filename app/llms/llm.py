from anthropic import Anthropic
import json
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()
MODEL = "claude-haiku-4-5"


def add_user_message(prompt: str, messages: list[dict]):
    user_message = {"role": "user", "content": prompt}
    messages.append(user_message)
    return messages


def add_assistant_message(prompt: str, messages: list[dict]):
    assistant_message = {"role": "assistant", "content": prompt}
    messages.append(assistant_message)
    return messages


def call_anthropic(messages: list[dict]):
    params = {"model": MODEL, "max_tokens": 300, "messages": messages}
    response = client.messages.create(**params)
    return response


if __name__ == "__main__":
    messages = []
    result = add_user_message("Hello! how are you?", messages)
    response = call_anthropic(messages)
    print("786 response ===>", response)
