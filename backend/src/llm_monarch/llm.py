from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name="distilgpt2"


# LLM Setup
# llm = "EleutherAI/gpt-neo-1.3B"
# Load LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

