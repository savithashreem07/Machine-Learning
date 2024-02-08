import openai
import os

from openai import OpenAI
from products import products
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

# Define OpenAI API_KEY
with open("/home/savitha07/.env") as env:
    for line in env:
        key, value = line.strip().split('=')
        os.environ[key] = value

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
)

# Define delimiter
delimiter = "####"

# Use text completion to generate the required content
def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=500):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens, 
    )
    return response.choices[0].message.content


# Step 1: Generate customer comment based on the product input 
def generate_customer_comment(products):

    system_message = f"""{products}"""
    user_message = f"""Generate comment in less than 100 words about the products"""

    messages =  [ 
    {'role':'system',
    'content': system_message},
    {'role':'user',
    'content': f"{delimiter}Assume you are a customer of the electronics company. {user_message}{delimiter}"},   
    ]

    comment = get_completion_from_messages(messages)
    print('Comment:\n', comment)
    return comment


# Step 2: Generate a subject for the email from the comment 
def generate_email_subject(comment):
    system_message = comment
    user_message = f"""Please generate a subject for the email from the comment using Inferring technique."""

    messages =  [  
    {'role':'system',
    'content': system_message},   
    {'role':'user',
    'content': f"{delimiter}Assume that you are a customer support representative of the electronics company. {user_message}{delimiter}"},  
    ]

    subject = get_completion_from_messages(messages)
    print('Subject of the email:\n', subject)
    return subject


# Step 3: Create a summary of the comment
def generate_summary(comment):
    system_message = comment
    user_message = f"""Provide a concise summary of the comment in at most 30 words."""

    messages =  [  
    {'role':'system',
    'content': system_message},   
    {'role':'user',
    'content': f"{delimiter}Assume that you are a customer support representative of the electronics company. {user_message}{delimiter}"},    
    ]

    summary = get_completion_from_messages(messages)
    print('Summary of the comment:\n', summary)
    return summary


# Step 4: Analyze the sentiment of the comment and tell if it is positive or negative
def analyze_sentiment(comment):
    system_message = comment
    user_message = f"""Do sentiment analysis of the comment using Inferring technique. Just mention if it is positive or negative in one word."""

    messages =  [  
    {'role':'system',
    'content': system_message},   
    {'role':'user',
    'content': f"{delimiter}Assume that you are a customer support representative of the electronics company. {user_message}{delimiter}"},    
    ]

    sentiment = get_completion_from_messages(messages)
    print('Sentiment of the comment:\n', sentiment)
    return sentiment


# Translate the given content into the selected language
def get_translation(email, language):
    system_message = email
    user_message = f"""Translate the given email content into {language} using Transforming technique"""

    messages =  [  
    {'role':'system', 
    'content': system_message},    
    {'role':'user', 
    'content': f"{delimiter}{user_message}{delimiter}"},  
    ]

    translate = get_completion_from_messages(messages)
    print(f"Translation of customer comment email in {language}: ")
    print(translate+"\n")
    return translate

# Step 5: Generate email based on the comment, summary, sentiment and subject generated
def generate_email(comment, subject, summary, sentiment):
    system_message = comment + subject + summary + sentiment
    user_message = f"""Create an email to be sent to the customer based on the {comment} and {sentiment}, including {subject}, {summary} in a proper format having subject and other content."""

    messages =  [  
    {'role':'system',
    'content': system_message},   
    {'role':'user',
    'content': f"{delimiter}Assume that you are a customer support representative of the electronics company. {user_message}{delimiter}"},    
    ]

    email = get_completion_from_messages(messages)
    print('Email generated:\n', email)
    return email


@app.route("/", methods=("GET", "POST"))
def index():
    comment = None
    language = 'en'
    email = None

    if request.method == "POST":
        language = request.form.get("language")
        translate_comment = request.form.get("translate-comment")
        translate_email = request.form.get("translate-email")
        comment = generate_customer_comment(products)
        subject = generate_email_subject(comment)
        summary = generate_summary(comment)
        sentiment = analyze_sentiment(comment)
        email = generate_email(comment, subject, summary, sentiment)

        if translate_email:
            email = get_translation(email, language)

        if translate_comment:
            comment = get_translation(comment, language)
    
    return render_template('index.html', comment = comment, language = language, email = email)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)