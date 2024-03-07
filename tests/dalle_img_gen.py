from openai import OpenAI
import requests 

client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="pouring water on a toaster",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)

# url='https://oaidalleapiprodscus.blob.core.windows.net/private/org-5uJ95hOqrsIpjsYtuwgkwJ3x/user-9nuFhw4BQVVznd85On41n9nJ/img-YQcqUNePOpyHg91PixthH9dF.png?st=2024-02-09T15%3A34%3A05Z&se=2024-02-09T17%3A34%3A05Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-02-08T21%3A17%3A34Z&ske=2024-02-09T21%3A17%3A34Z&sks=b&skv=2021-08-06&sig=6y5VMCZoncIntyhXoV752IRrlKJ59MuvS%2B5TxLde2c4%3D'

data = requests.get(image_url).content 

f = open("/home/saumyas/Projects/VLN-CE-Plan/tests/media/unsafe5.png",'wb') 
f.write(data) 
f.close()