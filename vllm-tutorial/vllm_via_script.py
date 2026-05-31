from vllm import LLM, SamplingParams


llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", gpu_memory_utilization=0.7)

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    max_tokens=100,
)


output = llm.generate(
    ["What first 3 orders of time derivative of position called?"], params)

print(output[0].outputs[0].text.strip())
