from anthropic import Anthropic

client = Anthropic()
msg = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print(msg.content[0].text)
