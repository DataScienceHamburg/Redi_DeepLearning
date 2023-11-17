#%% packages
import openai
from openai import OpenAI
openai.api_key = os.getenv('OPENAIAPI')
client = OpenAI()
import os
# %% load environment variable OPENAIAPI
# %%
text_to_translate = 'Hello, my name is John and I am a software engineer.'
prompt = f"Please translate from english to german the text inside <<>>. <<{text_to_translate}>>"

# %%
completion = client.chat.completions.create(model='gpt-3.5-turbo', messages=[{"role": "user", "content": prompt}])

# %%
completion.choices[0].message.content
# output: '<<Hallo, mein Name ist John und ich bin ein Softwareingenieur.>>'
# %%
