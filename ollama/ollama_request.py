import json
import requests

def parse_message(message:str):
    return [{"role": "user", "content": message}]

model_name = "llama3.2"
url = f"http://127.0.0.1:11434/api/chat"

message = "Can you make a bullet point list of the steps for build a wooden house"

# request JSON
request = {
    "model": model_name,
    "messages": parse_message(message)
}

# Send request
respone = requests.post(url, json=request, stream=True)


# Parse respone
if respone.status_code == 200: # OK
    print("Successfully recieved respone from ollama server")
    for line in respone.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                print(json_data["message"]["content"], end="")
            except json.JSONDecodeError:
                print("Failed to decode json data")
    print()
else:
    print(respone.text)
    raise requests.HTTPError(f"Recieved respone code {respone.status_code} from server, expected code 200 (OK)")