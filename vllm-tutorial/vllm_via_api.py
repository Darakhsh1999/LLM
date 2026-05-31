from openai import OpenAI


# To spin up a model run this in bash
# vllm serve "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --gpu-memory-utilization 0.7

client = OpenAI(base_url="http://localhost:8000/v1", api_key="arashapikey")


response = client.chat.completions.create(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    messages=[
        {"role": "system", "content": "You are a P.hd student in physics with a lot of experience in classical mechanics"},
        {"role": "user", "content": "What are the names of the time derivatives of position $x$ called in classical mechanics? What is the 3rd order time derivate called? Give me the names only and not the symbol."}
    ],
    max_tokens=300,
    temperature=0.7,
)

print(response.choices[0].message.content)