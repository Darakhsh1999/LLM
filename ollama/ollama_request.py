""" """

import os
import json
import utils
import requests

from dotenv import load_dotenv
from config import Config

load_dotenv()
c = Config()
url = f"{os.getenv("OLLAMA_URL")}/api/chat"

message = "Can you make a bullet point list of the steps for build a wooden house"

# request JSON
request = {
    "model": c.model_name,
    "messages": utils.parse_message(message)
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