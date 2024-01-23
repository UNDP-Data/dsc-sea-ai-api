import openai

# use this function to make simple openAI Calls
def callOpenAI(prompt, openai_deployment):  
    response_entities = openai.chat.completions.create(
                    model=openai_deployment,
                    temperature=0,
                    messages=[
                        {"role": "user", "content": prompt},
                    ]
                )
    response = response_entities.choices[0].message.content
    return response
