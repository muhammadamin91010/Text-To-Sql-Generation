# Text to SQL Generator

This repository contains a Gradio-based app for generating SQL queries from input text queries using fine-tuned models. It includes five fine-tuned models for SQL generation based on the **Qwen2.5** and **Llama3.2** architectures.

The interface allows users to input natural language queries and receive clean, formatted SQL queries as output. Users can select one of the five models to perform SQL generation.

---

## How to Run the App Locally

### Requirements
It is recommended to create a new conda environment to avoid dependency issues.

1. **Create a new conda environment:**
   ```bash
   conda create --name sql_gen_env python=3.10
   conda activate sql_gen_env
   ```

2. **Install the required packages:** Make sure to install all dependencies listed in the requirements.txt file.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Gradio app:** After installing the dependencies, you can run the run_sql_generation_gradio_interface.py script to launch the app.

   ```bash
   python run_sql_generation_gradio_interface.py
   ```
   
On the first run, the app will download the selected models from Hugging Face. On subsequent runs, it will load the models locally. 
These models are openly available on Hugging Face at the following repositories:

- [QWen2.5-0.5B](https://huggingface.co/abdulmannan-01/qwen-2.5-0.5b-finetuned-for-sql-generation)
- [QWen2.5-1.5B](https://huggingface.co/abdulmannan-01/qwen-2.5-1.5b-finetuned-for-sql-generation)
- [QWen2.5-3B](https://huggingface.co/abdulmannan-01/qwen-2.5-3b-finetuned-for-sql-generation)
- [Llama3.2-1B](https://huggingface.co/abdulmannan-01/Llama-3.2-1b-finetuned-for-sql-generation)
- [Llama3.2-3B](https://huggingface.co/abdulmannan-01/Llama-3.2-3b-finetuned-for-sql-generation)

## Example Input Queries

Here are five example queries ranging from easy to complex that you can use to test the app:

- **Easy**: List all customers who live in New York.
- **Moderate**: Find all employees who joined the company before 2020.
- **Moderate**: Show the total sales for each product category.
- **Hard**: List all customers who purchased both 'laptops' and 'smartphones'.
- **Hard**: Find the top 5 customers who spent the most money on orders placed in the last year, grouped by customer and ordered by the total spent.

---

## Running the App on Colab

You can also run this app on Colab using the `sql_generation_gradio_interface_colab.ipynb` notebook included in the repo. Simply open the notebook in Colab, install the necessary dependencies, and run the cells to launch the Gradio app in the Colab environment.
