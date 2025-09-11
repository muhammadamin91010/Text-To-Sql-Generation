import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sqlparse

models = {
    "QWen2.5-0.5B": ("abdulmannan-01/qwen-2.5-0.5b-finetuned-for-sql-generation", "abdulmannan-01/qwen-2.5-0.5b-finetuned-for-sql-generation"),
    "QWen2.5-1.5B": ("abdulmannan-01/qwen-2.5-1.5b-finetuned-for-sql-generation", "abdulmannan-01/qwen-2.5-1.5b-finetuned-for-sql-generation"),
    "QWen2.5-3B": ("abdulmannan-01/qwen-2.5-3b-finetuned-for-sql-generation", "abdulmannan-01/qwen-2.5-3b-finetuned-for-sql-generation"),
    "Llama3.2-1B": ("abdulmannan-01/Llama-3.2-1b-finetuned-for-sql-generation", "abdulmannan-01/Llama-3.2-1b-finetuned-for-sql-generation"),
    "Llama3.2-3B": ("abdulmannan-01/Llama-3.2-3b-finetuned-for-sql-generation", "abdulmannan-01/Llama-3.2-3b-finetuned-for-sql-generation")
}


loaded_models = {}

def load_model(model_key):
    model_name, tokenizer_name = models[model_key]
    if model_key not in loaded_models:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        loaded_models[model_key] = (model, tokenizer)
    return loaded_models[model_key]


def generate_sql(query, model_name):
    model, tokenizer = load_model(model_name)

    prompt = '<|im_start|>system\nYou are a MySQL SQL Writer. You must generate clean SQL statements using MySQL Syntax.<|im_end|>\n'
    prompt += f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n'

    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output[0], skip_special_tokens=False)
    first_semicolon = output_text.find(';')
    if first_semicolon != -1:
        output_text = output_text[:first_semicolon + 1]

    assistant_start = '<|im_start|>assistant\n'
    assistant_end = '<|im_end|>'
    start_idx = output_text.find(assistant_start) + len(assistant_start)
    end_idx = output_text.find(assistant_end, start_idx)

    if end_idx == -1:
        end_idx = len(output_text)

    generated_sql = output_text[start_idx:end_idx].strip()
    formatted_query = sqlparse.format(generated_sql, reindent=True)
    
    return formatted_query


def sql_generator(query, model_choice):
    return generate_sql(query, model_choice)

interface = gr.Interface(
    fn=sql_generator,
    inputs=[
        gr.Textbox(lines=5, label="Input Text Query"),  
        gr.Radio(list(models.keys()), label="Choose a Model")  
    ],
    outputs=gr.Textbox(label="Generated SQL Query"),  
    title="Text to SQL Generator",
    description="Generate SQL queries from natural language text using fine-tuned models. Select a model from the options below."
)

interface.launch(share=True)