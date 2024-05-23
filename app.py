
import streamlit as st
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,pipeline
import json
import torch

def getLLAMAresponse(input_text, no_words, blog_style):
    config_data = json.load(open("config.json"))
    huggingface_token = config_data["HUGGINGFACE_TOKEN"]
    #Quantization
    quant_config = BitsAndBytesConfig(
                                      load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",token=huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",device_map="auto",quantization_config=quant_config,
                                                token=huggingface_token)
    template = """Write a definition for {blog_style} job profile for a topic {input_text} within {no_words} words."""
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"], template = template)
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)

    text_generator = pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=128)
    response = text_generator(formatted_prompt)
    generated_text = response[0]["generated_text"]
    return(generated_text)

# Design the web page

st.set_page_config(page_title="Generate Blogs",
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Article")

input_text = st.text_input("Enter the blog topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.number_input('No. of words',min_value=1, step=1, value=50,format="%d")

with col2:
    blog_style = st.selectbox('Write article for', ('Researchers', 'Data Scientist'), index=0)

submit = st.button("Generate")

if submit:
    response = getLLAMAresponse(input_text,no_words,blog_style)
    st.write(response)
