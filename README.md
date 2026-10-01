# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 Midterm Project — Part 1: Dataset Collection

## Steps I Completed

### 1. Set Up the Project Environment

I created a Python virtual environment and installed the required dependencies using:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generated the Prompts

I ran:

```bash
python build_prompts.py
```

This generated a dataset of **120 original prompts**, divided equally across four categories:

* 30 coding prompts
* 30 math prompts
* 30 factual prompts
* 30 writing prompts

The 120 prompts were then divided into **20 batches of 6 prompts each** and saved in `data/batches/`.

### 3. Collected Responses from the LLMs

I collected responses from the free web chat interfaces of **Claude, ChatGPT, and Gemini**.

For each model, I:

1. Opened a **new chat for each batch** to prevent previous conversations from influencing the responses.
2. Turned off memory and custom instructions when those options were available.
3. Pasted the full contents of each batch file, starting with `batch_01.txt`.
4. Sent the six prompts to the chatbot.
5. Copied the chatbot's **entire response without editing or formatting it**.
6. Saved the response in the appropriate model folder, such as:

   * `data/raw/claude/batch_01.txt`
   * `data/raw/chatgpt/batch_01.txt`
   * `data/raw/gemini/batch_01.txt`
7. Repeated this process for all **20 batches** for each of the three models.

This resulted in approximately **20 messages/responses per chatbot**, with each response containing six prompts.

### 4. Preserved the Original Responses

I kept the chatbot responses unedited because formatting, wording, structure, and other characteristics of the original responses may be useful signals for identifying which LLM produced them.

If a response was incomplete or stopped before answering all of the prompts, I continued the conversation or reran the batch so that the complete response could be collected.

I also kept track of the model/version displayed by each chatbot because the free versions of these services can change over time.

### 5. Built the Final Dataset

After collecting the raw responses, I ran:

```bash
python parse_responses.py
```

This processed the individual response files and created:

```text
data/dataset.csv
```

The resulting dataset contains the following columns:

* `llm_name` — the model that generated the response
* `prompt_id` — the ID of the original prompt
* `category` — coding, math, factual, or writing
* `input` — the original prompt
* `output` — the LLM's response

The script also provided summary information, including the number of responses per model and category, average response length, and any missing responses that needed to be addressed.

## Dataset Information

The prompts consisted of **120 original questions** that I created for this project across four task types: coding, math, factual questions, and writing.

The responses were collected manually from the free web interfaces for **Claude, ChatGPT, and Gemini**.

The model versions displayed by the applications were:

* Claude: `sonnet-5.5`
* ChatGPT: `gpt-6-luna`
* Gemini: `gemini-3.5-Flash-Lite`

Each batch contained six prompts, and every batch was submitted in a fresh chat. The responses were saved in their original, unedited form.

## Limitations

There were several limitations to the data collection process:

* The dataset is relatively small.
* Using six prompts in a single message may produce slightly different responses than submitting each prompt individually.
* Free-tier chatbot services may change the underlying model or model version over time.
* The responses were collected manually, which introduces the possibility of human error during copying or organization.

No private or third-party data was included in the dataset.
