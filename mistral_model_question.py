from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
hf_token = "hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=hf_token
)

def generate_answer(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,#less creativity as it is quant focussed can go for 0.7
        top_p=0.9
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


query = "What is the key advantage of multivariate GARCH models over univariate ones?"
chunks = hybrid_retrieve(query)
prompt = build_prompt(query, chunks)
answer = generate_answer(prompt)

print("\nFinal Answer:\n")
print(answer)
