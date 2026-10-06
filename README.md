# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 Midterm Project
Part 1: Dataset Collection

## Part 1: Dataset Collection

### 1. Set Up the Project

I created a Python virtual environment and installed the required packages.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2. Created the Prompts

I ran:

```bash
python build_prompts.py
```

This created 120 prompts across four categories:

* 30 coding prompts
* 30 math prompts
* 30 factual prompts
* 30 writing prompts

The prompts were split into 20 batches with 6 prompts in each batch. The batches were saved in `data/batches/`.

### 3. Collected the LLM Responses

I collected responses from the free web versions of Claude, ChatGPT, and Gemini.

For each model, I used a new chat for every batch so that previous conversations would not affect the results. I also turned off memory and custom instructions when those options were available.

For each batch, I pasted the six prompts into the chatbot and copied the full response without changing the wording or formatting. The responses were saved in the corresponding model folder:

```text
data/raw/claude/batch_01.txt
data/raw/chatgpt/batch_01.txt
data/raw/gemini/batch_01.txt
```

I repeated this for all 20 batches for each model.

### 4. Kept the Original Responses

I kept the responses as they were given by the models because things like formatting, wording, structure, and response style could be useful when trying to identify the model.

If a response was incomplete, I reran the batch or continued the conversation to get a complete response. I also recorded the model/version shown by each website since these versions can change over time.

### 5. Created the Dataset

Once the responses were collected, I ran:

```bash
python parse_responses.py
```

This combined the individual response files into:

```text
data/dataset.csv
```

The dataset includes:

* `llm_name` - the model that generated the response
* `prompt_id` - the ID of the original prompt
* `category` - coding, math, factual, or writing
* `input` - the original prompt
* `output` - the model's response

The script also reports the number of responses for each model and category, average response length, and any missing responses.

## Part 2: Models and Experiments

The main experiments can be run with the following commands:

```bash
pip install -r requirements.txt

python baseline.py                              # TF-IDF + logistic regression

python train_nn.py --exp rq1 --quick            # quick smoke test
python train_nn.py --exp rq1                    # RQ1: CNN + BiLSTM
python train_nn.py --exp rq2                    # RQ2: input/output combinations
python train_nn.py --exp rq3                    # RQ3: leave-one-task-out
python train_nn.py --exp rq1 --strip_format     # RQ4: formatting ablation
python analyze_features.py                      # RQ4: stylometric features
python summarize.py                              # generate tables and confusion matrices
```

### Experiment Notes

* All experiments use the same 5-fold splits. The splits are prompt-grouped and task-stratified in `common.py`.
* `train_nn.py` uses 3 seeds across 5 folds. `--seeds 1` can be used to run a faster version.
* Results are saved in `results/` and can be used in `REPORT_TEMPLATE.md`.

## Dataset Information

The dataset contains 120 original prompts that I created for the project. They cover four types of tasks: coding, math, factual questions, and writing.

The responses were collected manually from the free web interfaces of Claude, ChatGPT, and Gemini.

The model versions displayed by the applications during collection were:

* Claude: `sonnet-5.5`
* ChatGPT: `gpt-6-luna`
* Gemini: `gemini-3.5-Flash-Lite`

Each batch contained six prompts and was submitted in a fresh chat. The responses were saved without editing them.

## Limitations

There are a few limitations to the dataset:

* The dataset is relatively small.
* Sending six prompts in one message may produce different results than sending each prompt separately.
* Free-tier chatbot models and versions can change over time.
* Since the responses were collected manually, copying or organization errors are possible.

No private or third-party data was included in the dataset.
