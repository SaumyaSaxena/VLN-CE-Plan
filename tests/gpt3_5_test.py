import openai
import json, os
import tiktoken

def response_gen(messages):
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    print(response)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


# messages=[
#     {"role": "system", "content": "You are navigating in an indoor environment. You are trying to reach a certain goal in a room given instructions."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
#     {"role": "user", "content": "Where was it played?"}
# ]

messages=[
    {"role": "system", "content": "You are trying to reach a certain goal given some instructions."},
    {"role": "user", "content": "Instruction: 'You are facing a bed with a teal throw on on it.  You are going to exit this bedroom.  You are going to step out into the hallway in front of a washer and dryer and make a right. You have now entered a living room and a kitchen area.  Once in this room you are going to veer to the left \
        and go towards the kitchen area.  You are going to stand next to the \
        rug in the kitchen area in front of the stove and then you are going to \
        head towards the sofa in the living area which will be right next to a \
        lamp. You are going to hop over the round ottoman going towards the \
        white door and once you are right next to the white door with a white \
        wicker chair on your right against the wall you are done.' Give output in the following format: \
        'goto <goal> which is near <nearest landmark>'"}
]

print('Number of tokens:',num_tokens_from_messages(messages, model="gpt-3.5-turbo"))
# print(response_gen(messages))