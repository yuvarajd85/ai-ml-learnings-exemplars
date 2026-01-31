# <u> <center> AI Agents Workkshop </u> </center>

---- 

## Day-1 01-24-2025

- **Prompt and Context Engineering for Agents**

    Learn how to design effective prompts, manage context, set goals and constraints, and improve agent reliability.

- **Create a RAG App using n8n**

    Build a Retrieval-Augmented Generation app using low-code workflows. Connect documents, embeddings, and LLMs to answer queries accurately.

- **Build Multi-Agent Systems in Langflow**

    Understand how multiple agents collaborate. Design and orchestrate specialized agents using visual workflows.

- **Agentic AI Architectures**

    Explore how agent-based systems are designed, including memory, tools, planning loops, and failure handling.


## Day-2 01-25-2024

- **Building Multi-Agent Workflows with AgentKit**

    Learn how to coordinate agents, manage task delegation, and build scalable, production-ready workflows.

- **Building AI Products with Lovable and Cursor**

    Rapidly prototype AI products, connect agents to user interfaces, and accelerate development using AI-powered tools.

- **AI Evals**

    Learn how to evaluate AI systems using accuracy, reliability, cost, and latency metrics, and design effective evaluation strategies.

-----

### Day-1 Notes

#### Frameworks for System prompt

- ICE
  - Instruction + Context + Example 
  - More like Few Shot Prompting
- RICE
  - Role + Instruction + Context + Example
- CRISP
  - Context + Role + Instruction + Specifics + Parameters
- CRISPE
  - Context + Role + Instruction + Specifics + Parameters + Evals 

These “frameworks” are mostly mnemonics for prompt structure (what to include + in what order). They’re not magical spells; they just reduce ambiguity and force you to specify the stuff that actually controls model behavior.

Also: “ICE” and “RICE” aren’t single canonical standards—different authors use the same acronym for slightly different fields. Example: ICE can mean Intent–Context–Expectation  ￼ or Instructions–Context–Examples–Task (ICE-T)  ￼. Same idea, different labels.

Quick map: frameworks → prompting techniques


| Framework | Stands for (common version) | Best for | Maps to techniques |
|---------|----------------------------|----------|--------------------|
| ICE | Intent, Context, Expectation | Simple, fast “make it do X” prompts | Instruction prompting + output constraints |
| ICE-T | Instructions, Context, Examples, Task | Reusable templates, automation | Few-shot + templating + task spec |
| RICE / RICE-Q | Role, Instructions, Context, Examples (+ Questions) | High-quality answers + interactive clarifying | Role prompting + few-shot + clarification step |
| CRISPE | Context, Role, Instructions, Steps, Parameters, Example | Complex tasks needing rigor | Decomposition (Steps) + constraints (Parameters) + few-shot |
| CRISPY | Context, Role, Instructions, Specifics, Parameters, Yielding | Marketing/content + explicit output format | Format forcing + constraints + iteration cue (“yielding”) |
| APE | Action, Purpose, Expectation | Beginner-friendly, compact prompts | Instruction + goal + output spec |
| RACE | Role, Action, Context, Explanation | “Do this and explain why” | Role prompting + rationale request |
| CARE | Context, Action, Result, Example | Case studies, postmortems | Structured writing + example anchoring |
| ROSES | (varies) Role, Objective, Steps, Example, Style | “Do X in a particular style” | Role + decomposition + few-shot + style control |
| COAST / CO-STAR | (varies; “Context/Objective/Actions/Style/Tone…”) | Creative + brand voice consistency | Style prompting + constraints + structure |
| CREATE | (varies; used as a multi-part structure) | Longer form generation | Multi-stage generation + format forcing |
| TAG | Task, Action, Goal | Quick operational prompts | Instruction + steps + objective |

If you want “all frameworks”: nobody has a definitive list because people invent acronyms weekly. The ones above are the most commonly referenced clusters in public writeups right now.  ￼

⸻

#### LangChain pattern (same for every framework)

All of these boil down to:
- a template string
- a dict of variables
- optional: system vs human message split

Below I’ll use ChatPromptTemplate.fromTemplate() (single-message). If you want stricter control, switch to fromMessages([("system", ...), ("human", ...)]).

```python
from langchain_core.prompts import ChatPromptTemplate
```



----

#### Framework-by-framework examples + LangChain code

#### 1) ICE (Intent–Context–Expectation)

Example prompt (human):

- Intent: summarize
- Context: doc + audience
- Expectation: bullets + max length

```python

from langchain_core.prompts import ChatPromptTemplate

ice_template = """Intent:
{intent}

Context:
{context}

Expectation (format, length, tone):
{expectation}
"""

ice_prompt = ChatPromptTemplate.fromTemplate(ice_template)

# usage:
messages = ice_prompt.format_messages(
    intent="Summarize the meeting notes",
    context="Audience: execs. Topic: Q4 risks. Notes:\n" + "{notes}",
    expectation="Return 6 bullets max. Include top 3 risks + mitigations. No fluff."
)
``` 

**Technique mapping:** instruction prompting + explicit output constraints.  ￼

⸻

#### 2) ICE-T (Instructions–Context–Examples–Task)

Best when you want a prompt you can reuse across many inputs.  ￼

```python

ice_t_template = """Instructions:
{instructions}

Context:
{context}

Examples:
{examples}

Task:
{task}

Input:
{input_text}
"""

ice_t_prompt = ChatPromptTemplate.fromTemplate(ice_t_template)

messages = ice_t_prompt.format_messages(
    instructions="Follow the examples' style exactly. Be concise and accurate.",
    context="You are classifying customer tickets for routing.",
    examples=(
        "Example 1:\nInput: 'Refund not received'\nOutput: Billing\n\n"
        "Example 2:\nInput: 'App crashes on login'\nOutput: Bug"
    ),
    task="Classify the input into one label: Billing, Bug, Feature, Account, Other.",
    input_text="Customer says: 'Can't change my email address'"
)

```
**Technique mapping:** few-shot prompting + templating + constrained classification.  ￼

⸻

#### 3) RICE / RICE-Q (Role–Instructions–Context–Examples–Questions)

The “Q” variant explicitly asks the model to ask clarifying questions when needed.  ￼

```python

rice_q_template = """Role:
{role}

Instructions:
{instructions}

Context:
{context}

Examples:
{examples}

Questions (ask if anything is ambiguous):
{questions}

User request:
{request}
"""

rice_q_prompt = ChatPromptTemplate.fromTemplate(rice_q_template)

messages = rice_q_prompt.format_messages(
    role="You are a senior data engineer reviewing a proposed ETL design.",
    instructions="Point out risks, propose fixes, and keep it practical.",
    context="Stack: AWS Glue, S3, Athena, Step Functions. SLA: 2 hours.",
    examples="If partitioning is missing, recommend partition keys and compaction.",
    questions="If critical info is missing, ask up to 3 questions before finalizing.",
    request="Review this design: we’ll dump JSON into S3 and query it directly in Athena."
)

```

**Technique mapping:** role prompting + critique prompting + clarification gate.  ￼

⸻

#### 4) CRISPE (Context–Role–Instructions–Steps–Parameters–Example)

Great when you need stepwise structure and hard constraints.  ￼

```python

crispe_template = """Context:
{context}

Role:
{role}

Instructions:
{instructions}

Steps:
{steps}

Parameters (hard constraints):
{parameters}

Example output:
{example_output}

Task input:
{input_text}
"""

crispe_prompt = ChatPromptTemplate.fromTemplate(crispe_template)

messages = crispe_prompt.format_messages(
    context="We are building a RAG chatbot for SEC filings.",
    role="You are an LLM application architect.",
    instructions="Design the chunking + retrieval strategy and justify it.",
    steps="1) Identify goals 2) Propose approach 3) Give parameter defaults 4) Risks/mitigations",
    parameters="Max response 250 words. Include 1 small table. No vendor fluff.",
    example_output="(Example)\nGoal: ...\nApproach: ...\nDefaults: ...\nRisks: ...",
    input_text="Docs are 10-K PDFs; we need Q&A with citations."
)

``` 
**Technique mapping:** decomposition (“Steps”) + constraint prompting (“Parameters”) + example anchoring.  ￼

⸻

#### 5) CRISPY (Context–Role–Instructions–Specifics–Parameters–Yielding)

This is basically CRISPE but tuned toward output “yielding” (deliverable formatting and iteration).  ￼

```python

crispy_template = """Context:
{context}

Role:
{role}

Instructions:
{instructions}

Specifics (what to include/avoid):
{specifics}

Parameters (format + limits):
{parameters}

Yielding (how to deliver):
{yielding}

Input:
{input_text}
"""

crispy_prompt = ChatPromptTemplate.fromTemplate(crispy_template)

messages = crispy_prompt.format_messages(
    context="You are writing a launch email for an internal tool.",
    role="You are a product marketer for an enterprise audience.",
    instructions="Write the email body only.",
    specifics="Include: problem, 3 benefits, 1 CTA. Avoid hype words.",
    parameters="120-160 words. 6th-grade readability. No emojis.",
    yielding="Return exactly: Subject line + Body, separated by a blank line.",
    input_text="Tool: 'OpsLens' reduces on-call noise by clustering alerts."
)
```
**Technique mapping:** format forcing + constraints + delivery contract.  ￼

⸻

#### 6) APE (Action–Purpose–Expectation)

Compact and effective.  ￼

```python
ape_template = """Action:
{action}

Purpose:
{purpose}

Expectation:
{expectation}

Input:
{input_text}
"""

ape_prompt = ChatPromptTemplate.fromTemplate(ape_template)

messages = ape_prompt.format_messages(
    action="Draft a technical design doc outline",
    purpose="Align the team on scope and risks for a new data pipeline",
    expectation="Use headings + bullets. Include: goals, non-goals, architecture, risks.",
    input_text="Pipeline: ingest Kafka -> S3 -> Iceberg -> Athena. SLA 1 hour."
)
``` 

**Technique mapping:** instruction + goal + output spec.  ￼

⸻

#### 7) RACE (Role–Action–Context–Explanation)

Good when you want an answer plus justification.  ￼

```python

race_template = """Role: {role}
Action: {action}
Context: {context}
Explanation: {explanation}

Input:
{input_text}
"""

race_prompt = ChatPromptTemplate.fromTemplate(race_template)

messages = race_prompt.format_messages(
    role="You are a security engineer.",
    action="Review this API design for risks and propose mitigations.",
    context="Public-facing REST API. Auth via OAuth2. Logs in CloudWatch.",
    explanation="For each risk, explain impact + fix in 2 lines.",
    input_text="Design: client sends JWT in query string; we store full request logs."
)
```

**Technique mapping:** role prompting + critique prompting + “explain why”.  ￼

⸻

#### 8) CARE (Context–Action–Result–Example)

Useful for reporting, retros, and narrative that still stays concrete.  ￼

```python

care_template = """Context:
{context}

Action:
{action}

Result:
{result}

Example:
{example}

Write-up request:
{request}
"""

care_prompt = ChatPromptTemplate.fromTemplate(care_template)

messages = care_prompt.format_messages(
    context="Legacy batch jobs routinely missed SLA.",
    action="We migrated transformations to Spark + added data quality checks.",
    result="SLA met 98% of days; incidents dropped.",
    example="Example: P95 runtime dropped from 110m to 42m after partition tuning.",
    request="Write a 1-page postmortem summary for leadership."
)
```

**Technique mapping:** structured narrative + evidence anchoring.  ￼

⸻

#### 9) ROSES (Role–Objective–Steps–Example–Style)

There are multiple ROSES variants in the wild, but this is the common “do X in style Y” structure.  ￼

```python

roses_template = """Role:
{role}

Objective:
{objective}

Steps:
{steps}

Example:
{example}

Style:
{style}

Input:
{input_text}
"""

roses_prompt = ChatPromptTemplate.fromTemplate(roses_template)

messages = roses_prompt.format_messages(
    role="You are an AI tutor for backend engineers.",
    objective="Explain vector embeddings clearly without math overload.",
    steps="Start with intuition, then 1 simple analogy, then 1 practical example.",
    example="Analogy example: 'GPS coordinates for meaning'.",
    style="Friendly but no fluff. 250 words max.",
    input_text="Topic: cosine similarity vs dot product."
)
```
**Technique mapping:** role prompting + decomposition + style constraints + analogy prompting.  ￼

⸻

#### 10) TAG (Task–Action–Goal)

Quick operational prompts; good for agents and pipelines.  ￼

```python

tag_template = """Task:
{task}

Action:
{action}

Goal:
{goal}

Input:
{input_text}
"""

tag_prompt = ChatPromptTemplate.fromTemplate(tag_template)

messages = tag_prompt.format_messages(
    task="Normalize and validate user-provided addresses",
    action="Standardize abbreviations, validate ZIP, flag missing fields",
    goal="Return a clean JSON object plus an 'issues' array",
    input_text="Input address: '12 main st, philadelphia pa 19'"
)
```
**Technique mapping:** instruction + validation + output schema prompting.  ￼

⸻

#### 11) CREATE (variant)

CREATE is referenced as a multi-part content structure in “prompt frameworks” lists, but definitions vary a lot across sources.  ￼
A safe, reusable implementation is: Context → Request → Examples → Acceptance criteria → Tone → Extras.

```python

create_template = """Context:
{context}

Request:
{request}

Examples (optional):
{examples}

Acceptance criteria:
{criteria}

Tone:
{tone}

Extras / constraints:
{extras}
"""

create_prompt = ChatPromptTemplate.fromTemplate(create_template)

messages = create_prompt.format_messages(
    context="We’re proposing an MLOps rollout to a risk-averse enterprise.",
    request="Draft an executive summary slide (text only).",
    examples="(None)",
    criteria="Must include: business value, risk controls, timeline, cost bands.",
    tone="Crisp, executive, non-technical.",
    extras="Max 120 words. No buzzwords."
)
```
**Technique mapping:** acceptance-test prompting + tone control + constraints.  ￼

⸻
```shell
export ASTRA_DB_API_ENDPOINT="YOUR_API_ENDPOINT"
export ASTRA_DB_APPLICATION_TOKEN="YOUR_TOKEN"
```

```python
from astrapy.info import (
    CollectionDefinition,
    CollectionVectorOptions,
    VectorServiceOptions,
)
from astrapy.constants import VectorMetric

collection = database.create_collection(
    "COLLECTION_NAME",
    definition=CollectionDefinition(
        vector=CollectionVectorOptions(
            dimension=1536,
            metric=VectorMetric.DOT_PRODUCT,
            service=VectorServiceOptions(
                provider="openai",
                model_name="text-embedding-3-small",
                authentication={
                    "providerKey": "API_KEY_NAME",
                },
            ),
        ),
    ),
)
```

**Langflow vector db token**



----

### Day-2 Notes

- Lovable
  - Product creation
- Perplexity 

#### Eval 

- Accuracy
- Relevance
- Coherence
- Perplexity
- Faithfulness
- Bleu
- Rogue
- Gisting
- Toxicity
- Latency
- Hallucinate

#### Eval Techniques

- LLM as Judge
- Human in the loop 

-----

#### Topics to cover / Read

- ***Google Anti-Gravity***
- ***Windsurf***
- ***Agent Kit***


#### Important URLS

- https://karpathy.bearblog.dev/vibe-coding-menugen/
- https://www.menugen.app/
- https://n8n.io/workflows/5010-rag-starter-template-using-simple-vector-stores-form-trigger-and-openai/

