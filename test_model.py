from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained("./gpt2_finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_finetuned")

# Explicitly set pad_token_id if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

# Input text for generation
input_text = "The meaning of life is" 

# Tokenize input text and provide attention mask
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,  # Explicitly provide attention_mask
    max_length=200,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id
)

# Decode and print the output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

