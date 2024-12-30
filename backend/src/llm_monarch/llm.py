# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "EleutherAI/gpt-neo-1.3B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# prompt = "What is AI?"
# inputs = tokenizer(prompt, return_tensors="pt")

# # Generate a response
# outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)

# # Decode and print the response
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))