
model_name = "llama2"
param = ("temperature", 1)
system_prompt = "You are the character Spider-man from the marvel series. Answer as Spider-man only."

with open("Modelfile","w") as f:
    f.write(f"FROM {model_name} \n") 
    f.write(f"PARAMETER {param[0]} {param[1]} \n") 
    f.write(f'SYSTEM {system_prompt} \n') 