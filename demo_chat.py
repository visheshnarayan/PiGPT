import openai

print("Terminal based Chat GPT")

model="text-davinci-003"
openai.api_key="KEY HERE"
prompt=input("User question: ")

max_tokens=1024

response = openai.Completion.create(
    engine=model,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=0.5,
    top_p=1
)

print(f"\nChapGPT: {response.choices[0].text}")