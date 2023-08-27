## This notebook shows how to finetune OpenAI GPT3.5 Model -with 1 command- on the Mental Health Chat Dataset from HugginFace [here](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset) 

**Note:** The process is simple and OpenAI charges only for the amount of tokens that will pass through the model in training.
Although the cost of using the finetuned model is 10x the cost of the GPT3.5 and 1/3 the cost of GPT 4. 
Finetuning should be the last option in any project!

### Step 0: Import all Libraries and set API keys


```python
from collections import defaultdict
import os
import json
import numpy as np
import pandas as pd
import tiktoken
import openai

openai.api_key = "sk-xxxxx"
```

### Step 1: Download Dataset and Format It

- You can download the dataset directly from the Hub but it is a small dataset so i will just download the parquet file and import with pandas
- Originally it has one column but i will add a second one that will be fed to openai


```python
df = pd.read_parquet('train-mental.parquet')
df['text_openai_formated'] = None
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>text_openai_formated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>&lt;HUMAN&gt;: What is a panic attack?\n&lt;ASSISTANT&gt;:...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>&lt;HUMAN&gt;: What are symptoms of panic attack vs....</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>&lt;HUMAN&gt;: What are the types of Mental Illness?...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>&lt;HUMAN&gt;: What does mental-illness mean?\n&lt;ASSI...</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>&lt;HUMAN&gt;: How can you treat mental illness?\n&lt;A...</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check the biggest length of prompts
n_max_words = 0
for index, row in df.iterrows():
    n_max_words = max(n_max_words, len(row['text'].split()))

print(f"The longest string contains {n_max_words} words approximately {round(n_max_words/0.7)} tokens.")  
print("GPT 3.5 has a context length of 4096 tokens so no problem here")
```

    The longest string contains 369 words approximately 527 tokens.
    GPT 3.5 has a context length of 4096 tokens so no problem here


### System Message 

- The system prompt works as a guide to what our model want to talk and act like


```python
sys_msg = "DoctorGPT is a factual chatbot that is an expert on mental health. He's mission is to help the user all the time and never refuse to answer."
print(sys_msg)
```

    DoctorGPT is a factual chatbot that is an expert on mental health. He's mission is to help the user all the time and never refuse to answer.


### Transform the Data into the format OpenAI requires

<code>{"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": User_Input}, {"role": "assistant", "content": Model_Outout}]} </code>``

<code>{"messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": User_Input}, {"role": "assistant", "content": Model_Outout}]} </code>

### Data Transformation


```python
data_list = []
for index, row in df.iterrows():
    text = row['text']

    assistant_keyword = "<ASSISTANT>: "
    human_keyword = "<HUMAN>: "

    # Find the index of the keyword in the text
    idx = text.index(assistant_keyword)

    human_msg = text[:idx].replace(human_keyword, '')
    assistant_msg = text[idx:].replace(assistant_keyword, '')

    data = {"messages": [{"role": "system", "content": sys_msg}, 
                                                             {"role": "user", "content": human_msg}, 
                                                             {"role": "assistant", "content": assistant_msg}]}
    df.at[index, 'text_openai_formated'] = data
    data_list.append(data)
```

### Save the Transformed Data in a JSON file


```python
# Write each element of the list on a separate line in the JSON file
with open('test.json', 'w') as json_file:
    for item in data_list:
        json_file.write(json.dumps(item) + '\n')
```


```python

```

### OpenAI Script to validate the data and provide Cost Estimation


```python
# Next, we specify the data path and open the JSONL file
data_path = "test.json"

# Load dataset
with open(data_path) as f:
    dataset = [json.loads(line) for line in f]

# We can inspect the data quickly by checking the number of examples and the first item

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)

# Now that we have a sense of the data, we need to go through all the different examples and check to make sure the formatting is correct and matches the Chat completions message structure

# Format error checks
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant"):
            format_errors["unrecognized_role"] += 1

        content = message.get("content", None)
        if not content or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")

# Beyond the structure of the message, we also need to ensure that the length does not exceed the 4096 token limit.

# Token counting functions
encoding = tiktoken.get_encoding("cl100k_base")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    print(f"\n#### Distribution of {name}:")
    print(f"min / max: {min(values)}, {max(values)}")
    print(f"mean / median: {np.mean(values)}, {np.median(values)}")
    print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

# Last, we can look at the results of the different formatting operations before proceeding with creating a fine-tuning job:

# Warnings and tokens counts
n_missing_system = 0
n_missing_user = 0
n_messages = []
convo_lens = []
assistant_message_lens = []

for ex in dataset:
    messages = ex["messages"]
    if not any(message["role"] == "system" for message in messages):
        n_missing_system += 1
    if not any(message["role"] == "user" for message in messages):
        n_missing_user += 1
    n_messages.append(len(messages))
    convo_lens.append(num_tokens_from_messages(messages))
    assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

print("Num examples missing system message:", n_missing_system)
print("Num examples missing user message:", n_missing_user)
print_distribution(n_messages, "num_messages_per_example")
print_distribution(convo_lens, "num_total_tokens_per_example")
print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
n_too_long = sum(l > 4096 for l in convo_lens)
print(f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning")

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096

MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
TARGET_EPOCHS = 3
MIN_EPOCHS = 1
MAX_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens)
print(f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training")
print(f"By default, you'll train for {n_epochs} epochs on this dataset")
print(f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens")
print("See pricing page to estimate total costs")
print(f"Cost: ~ {round(n_epochs * (n_billing_tokens_in_dataset/1000) * 0.008, 2)} $")
```

    Num examples: 172
    First example:
    {'role': 'system', 'content': "DoctorGPT is a factual chatbot that is an expert on mental health. He's mission is to help the user all the time and never refuse to answer."}
    {'role': 'user', 'content': 'What is a panic attack?\n'}
    {'role': 'assistant', 'content': 'Panic attacks come on suddenly and involve intense and often overwhelming fear. They’re accompanied by very challenging physical symptoms, like a racing heartbeat, shortness of breath, or nausea. Unexpected panic attacks occur without an obvious cause. Expected panic attacks are cued by external stressors, like phobias. Panic attacks can happen to anyone, but having more than one may be a sign of panic disorder, a mental health condition characterized by sudden and repeated panic attacks.'}
    No errors found
    Num examples missing system message: 0
    Num examples missing user message: 0
    
    #### Distribution of num_messages_per_example:
    min / max: 3, 3
    mean / median: 3.0, 3.0
    p5 / p95: 3.0, 3.0
    
    #### Distribution of num_total_tokens_per_example:
    min / max: 83, 528
    mean / median: 248.26162790697674, 235.5
    p5 / p95: 130.3, 385.9
    
    #### Distribution of num_assistant_tokens_per_example:
    min / max: 29, 469
    mean / median: 189.02906976744185, 178.0
    p5 / p95: 73.2, 325.8
    
    0 examples may be over the 4096 token limit, they will be truncated during fine-tuning
    Dataset has ~42701 tokens that will be charged for during training
    By default, you'll train for 3 epochs on this dataset
    By default, you'll be charged for ~128103 tokens
    See pricing page to estimate total costs
    Cost: ~ 1.02 $


## Step 2: Upload Dataset


```python
openai.File.create(
  file=open("test.json", "rb"),
  purpose='fine-tune'
)
```




    <File file id=file-Cj0rMqbIvuSAxdGUAW4iqfU6 at 0x7f3f88a9a840> JSON: {
      "object": "file",
      "id": "file-Cj0rMqbIvuSAxdGUAW4iqfU6",
      "purpose": "fine-tune",
      "filename": "file",
      "bytes": 231260,
      "created_at": 1693155705,
      "status": "uploaded",
      "status_details": null
    }



## Step 3: Fine Tune Model


```python
# Uncomment the line below 

#openai.FineTuningJob.create(training_file="file-px68T9xYyYCZe3aFfZXCixIz", model="gpt-3.5-turbo")
```

<pre>

<FineTuningJob fine_tuning.job id=ftjob-xxxx at 0x7f9b7ea5ac50> JSON: {
  "object": "fine_tuning.job",
  "id": "ftjob-xxxx",
  "model": "gpt-3.5-turbo-0613",
  "created_at": 1692968797,
  "finished_at": null,
  "fine_tuned_model": null,
  "organization_id": "org-xxxxx",
  "result_files": [],
  "status": "created",
  "validation_file": null,
  "training_file": "file-px68T9xYyYCZe3aFfZXCixIz",
  "hyperparameters": {
    "n_epochs": 3
  },
  "trained_tokens": null

</pre>

Now we wait. This job will take less than 10 minutes but required time can vary a lot !


```python
# List 10 fine-tuning jobs
# print(openai.FineTuningJob.list(limit=10))
```

<pre>
{
  "object": "list",
  "data": [
    {
      "object": "fine_tuning.job",
      "id": "ftjob-QnlC2wyVRTs3RHgtvvmAF8na",
      "model": "gpt-3.5-turbo-0613",
      "created_at": 1692968797,
      "finished_at": 1692969848,
      "fine_tuned_model": "ft:gpt-3.5-turbo-0613:personal::7rR5CkYd",
      "organization_id": "org-XhR0hBmBoQbi9SrGpldtXXkJ",
      "result_files": [
        "file-CJmiIUHofIxBmwLLE7OIuaOE"
      ],
      "status": "succeeded",
      "validation_file": null,
      "training_file": "file-px68T9xYyYCZe3aFfZXCixIz",
      "hyperparameters": {
        "n_epochs": 3
      },
      "trained_tokens": 127071
    }
  ],
  "has_more": false
} 
</pre>

### You can check the progress by running the cell bellow


```python
# List up to 10 events from a fine-tuning job
openai.FineTuningJob.list_events(id="ftjob-xxxx", limit=2)

```




    <OpenAIObject list at 0x7f3f651c8a90> JSON: {
      "object": "list",
      "data": [
        {
          "object": "fine_tuning.job.event",
          "id": "ftevent-rVMzY4UlM9l8QjsHWcjd5cj9",
          "created_at": 1692969849,
          "level": "info",
          "message": "Fine-tuning job successfully completed",
          "data": null,
          "type": "message"
        },
        {
          "object": "fine_tuning.job.event",
          "id": "ftevent-L95faZTJY42wKQENj17ox9Of",
          "created_at": 1692969846,
          "level": "info",
          "message": "New fine-tuned model created: ft:gpt-3.5-turbo-0613:personal::7rR5CkYd",
          "data": null,
          "type": "message"
        }
      ],
      "has_more": true
    }



## Step 4 Use the model

### Once you see `"message": "Fine-tuning job successfully completed"` you are ready to use the model


```python
completion = openai.ChatCompletion.create(
  model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```


```python

```

### Extra Tips

- Cancel a job
- Delete a Model


```python
# Cancel a job
openai.FineTuningJob.cancel("ft-abc123")

openai.Model.delete("ft-abc123")
```


```python

```

### Check my Github for more: [here](https://github.com/AlexTs10)

### Are you looking for an AI Developer ?? contact here -> alextoska1010@protonmail.com


```python

```
