from typing import Optional
from groq import Groq
import requests,json,os
from dotenv import load_dotenv
load_dotenv()
# Initialize Groq client with API key from env
def get_groq_client():
    return Groq(api_key=os.getenv("GROK_KEY"))


def generate_answer(prompt: str, context: str, intent: Optional[str]) -> str:
    client = get_groq_client()
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant that must ONLY answer from the provided context. "
                    "If the answer is not explicitly present in the context, reply with 'Not found in context'. "
                    "Do not add explanations, domain knowledge, or assumptions beyond what is written in the context. "
                    "Your output should be strictly based on exact wording or paraphrases of the context."
                    "If the {intent} is understand_doc then try to give a concise summary or deatils using meta data and content."
                ),
            },
            {"role": "user", "content": f"Question: {prompt}\n\nContext:\n{context}"},
        ],
        temperature=0.2,
    )
    if completion.choices[0].message.content and completion.choices[0].message.content != "":
        return completion.choices[0].message.content
    else:
        return "Unexpected Server Error"


def generate_conv_title(
    api_key, model="llama-3.1-8b-instant", messages=None, temperature=0.2
):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Append a system prompt to instruct the model
    prompt_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates short, relevant titles for user queries. Respond only with the title.",
        }
    ] + messages

    payload = {"model": model, "messages": prompt_messages, "temperature": temperature}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"Groq API error {response.status_code}: {response.text}")

def extract_intent(response_text):
    try:
        content_str = response_text["choices"][0]["message"]["content"]
        data = json.loads(content_str)
        intent = data.get("intent", "unknown")
        confidence = data.get("confidence", 0.0)
        details = data.get("details", "")

        return intent, confidence, details

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print("Error parsing intent:", e)
        return "unknown", 0.0, "Failed to parse intent JSON"


def get_user_intent (api_key, model="llama-3.1-8b-instant", messages=None, temperature=0.2):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Append a system prompt to instruct the model
    prompt_messages = messages

    payload = {"model": model, "messages": prompt_messages, "temperature": temperature}

    response = requests.post(url, headers=headers, json=payload).json()
    intent, confidence, details = extract_intent(response)
    print(intent,"=========================")
    return intent


def call_groq_llm(
    api_key, model="llama-3.1-8b-instant", messages=None, temperature=0.7
):
    """
    Calls the Groq API to get a response from an LLM.

    Parameters:
        api_key (str): Your Groq API key.
        model (str): Model name (e.g., 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma-7b-it').
        messages (list): A list of message dicts in OpenAI format.
        temperature (float): Sampling temperature (0.0–1.0).

    Returns:
        str: The LLM-generated response.
    """
    if messages is None:
        messages = [{"role": "user", "content": "Hello, who are you?"}]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Groq API error {response.status_code}: {response.text}")
