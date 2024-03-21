import flask
from flask import Flask, render_template
import torch
import pickle
import transformers
from transformers import pipeline
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer 

app = flask.Flask(__name__)
memory = ConversationBufferMemory()

model = torch.load('llama2_model.pth')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = flask.request.form['message']
    
    memory.add_user_input(user_input)
    prompt = f"<s>[INST] {user_input} {memory.chat_history} [/INST]"

    # Use a pipeline for text generation if you have one
    result = pipe(prompt)  
    response = result[0]['generated_text']

    memory.add_bot_output(response) 
    return response

if __name__ == '__main__':
    app.run(debug=True) 
