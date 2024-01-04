# Compliance LLM

# Project description
The task is to build an API that does the following:

* Take a webpage as the input, and it has to check the content in the page against a compliance policy
* Return the findings (non-compliant results) in the response

As an example, we take

Stripe's public compliance policy: https://stripe.com/docs/treasury/marketing-treasury
Lets test it against https://www.joinguava.com/

## Prerequisite
 1. Python
 2. OpenAI API Key

## Installation
```commandline
pip install requirements.txt
```

## Usage
To start the server
```commandline
cd src
python server.py
```
Run http://localhost:5000/apidocs/#/default/post_api_v1_get_report

To test the application using Curl
```commandline
curl -X POST "http://localhost:5000/api/v1/get_report" -H  "accept: application/json" -H  "Content-Type: application/x-www-form-urlencoded" -d "url=https%3A%2F%2Fwww.joinguava.com%2F"
```