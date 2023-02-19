# imports
import openai

print("Terminal based ChatGPT using text-davinci-003::AI Club demo")

# ! variables 
model='text-davinci-003'
openai.api_key='KEY HERE'
max_tokens=1024

while True:
    try:
        prompt=input('User question: ')
        
        # * request
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=1
        )
        print(f"\nChatGPT: {response.choices[0].text}")

    except openai.error.RateLimitError:
        print("Our servers are full right now, ask again later!")

    if prompt=="END":
        break