from instructor import from_openai, Mode
from openai import OpenAI
from pydantic import BaseModel, Field

# Desired LLM output
class DataModel(BaseModel):
    name: str = Field(..., description="The name of the person")
    number_of_kids: int = Field(..., description="How many kids does the parent have?")

# Create a ollama client 
client = from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=Mode.JSON
)

respone = client.messages.create(
    model="llama3.1:latest",
    messages=[
        {
            "role": "user",
            "content": f"Adam has one son named Lukas and two daughters named Emma and Anna. Adam's wife is named Susan. Susan has a sister named Amanda."
        }
    ],
    response_model=DataModel
)

print(respone)
