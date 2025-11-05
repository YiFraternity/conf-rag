ANSWER_QUESTION_TEMPLETE = """{examples}{docs}Answer in the same format as before.Answer the following question in English by continuing the existing partial answer.

Instructions:
- Do NOT summarize.
- The "cur_answer" field should contain exactly the provided partial answer.
- The "continue" field should contain your newly generated continuation.
- Ensure logical and stylistic coherence with the existing text.
- Output strictly in valid JSON format (no extra text or comments outside JSON).

### Question:
{question}
### Current answer:
{gen_text}

### Output Format:
```json
{
  "cur_answer": "The existing partial answer here.",
  "continue": "Your continued and completed answer here."
}
```
"""

ANSWER_USE_DEMO_TEMPLATE = """Answer in the same format as before. """
ANSWER_NOT_USE_DEMO_TEMPLATE = """When the answer is sufficient to resolve the question, stop reasoning immediately and provide the answer, starting with "So, the answer is". """

ANSWER_USE_DOCS_TEMPLATE = """ Please step-by-step through the process of considering what kind of knowledge is needed to answer the question. If such knowledge is present in the document, please utilize it to answer the Question. If not, please disregard the docs."""


STEP_REASON_ANSWER_TEMPLATE = """{docs}
If the document contains information relevant to the question, please answer using a complete sentence based on that information. If the document does not contain content related to the current reasoning path, return the question.
- Question: {reasoning}
- Answer:"""

CONFIDENCE_VALUE_TEMPLATE = """Assess the provided response by evaluating its relevance to the Question and coherence with the History Response. Assign a confidence score between **0.0** (lowest) and **1.0** (highest), where values closer to 1.0 reflect stronger alignment and completeness. Prioritize reducing the score if evidence is insufficient or contradictions exist.

{docs}
### Question:
{question}
### History Response:
{history_resp}
### Your Response:
{response}
### Output Format:
```json
{{
    "Confidence": [0.0-1.0],
    "Current Sentence": [Sentence that requires confidence estimation],
    "Explanation": [Concise rationale]
}}
```"""

CONFIDENCE_LEVEL_TEMPLATE = """Evaluate the confidence level for the provided response **only with respect to the specific information it explicitly addresses**, even if the Question requests a comparison or involves multiple entities. Do **not** penalize the response for omitting information that has not yet been provided.

Present the numerical confidence level (1–5). Prioritize downward adjustments only when the **stated** information is ambiguous or unsupported — **not simply because the response is incomplete relative to the full Question**.

The evaluation should be strictly based on the information contained in the *current response only*, without inferring or relying on any additional context that has not yet been provided. **If the Question involves multiple entities or conditions, assess only the correctness of the part that is explicitly answered, and treat unaddressed components as pending rather than incorrect.** Any missing or supplementary information required for further evaluation will be provided in subsequent steps.

1. Very Uncertain
2. Not Certain
3. Lightly Certain
4. Fairly Certain
5. Very Certain

{docs}
### Question:
{question}
### History Response:
{history_resp}
### Your Response:
{response}
### Output Format:
```json
{{
    "Confidence": [1-5],
    "Current Sentence": [Sentence that requires confidence estimation],
    "Explanation": [Concise rationale]
}}
```"""

CONFIDENCE_VALUE_ONLY_CURR_RESPONSE_TEMPLATE = """Evaluate the confidence level for the provided response **only with respect to the specific information it explicitly addresses**, even if the Question requests a comparison or involves multiple entities. Do **not** penalize the response for omitting information that has not yet been provided.

Assess the provided response. Assign a confidence score between **0.0** (lowest) and **1.0** (highest), where values closer to 1.0 reflect stronger alignment and completeness. Prioritize downward adjustments only when the **stated** information is ambiguous or unsupported — **not simply because the response is incomplete relative to the full Question**.

The evaluation should be strictly based on the information contained in the *current response only*, without inferring or relying on any additional context that has not yet been provided. **If the Question involves multiple entities or conditions, assess only the correctness of the part that is explicitly answered, and treat unaddressed components as pending rather than incorrect.** Any missing or supplementary information required for further evaluation will be provided in subsequent steps.

{docs}
### History Response:
{history_resp}
### Your Response:
{response}
### Output Format:
```json
{{
    "Score": [0.0-1.0],
    "Explanation": [Concise rationale]
}}
```"""

CONFIDENCE_LEVEL_ONLY_CURR_RESPONSE_TEMPLATE = """Evaluate the confidence level for the provided response by analyzing its alignment with the Question and consistency with the History Response. Present the numerical confidence level (1–5). Prioritize downward adjustments in confidence scoring when supporting evidence is limited or ambiguous. The evaluation should be strictly based on the information contained in the *current response only*, without inferring or relying on any additional context that has not yet been provided. **If the Question involves multiple entities or conditions, assess only the part that is explicitly addressed in the current response, and do not speculate about any missing components.** Any missing or supplementary information required for further evaluation will be provided in subsequent steps.

1. Very Uncertain
2. Not Certain
3. Lightly Certain
4. Fairly Certain
5. Very Certain

{docs}
### History Response:
{history_resp}
### Your Response:
{response}
### Output Format:
```json
{{
    "Confidence Level": [1-5],
    "Explanation": [Concise rationale]
}}
```"""

KEYWORDS_TEMPLATE = """Please use 2 to 3 keywords to express the idea behind this sentence.
- Sentence: {sentence}"""

LACK_KNOW_INFO_TEMPLATE = """The following response contains hallucinations. Please generate query related to the last sentence of the response, based on both the question and the response, while using the entities from the question as much as possible.
- Question: What is the birthday of the 46th President of the United States?
- Response: Donald John Trump is the 46th President of the United States.
- Query: Who is the 46th President of the United States?
- Question: Which Golden Globe Lifetime Achievement Award did the person who played the protagonist in the movie Forrest Gump win?
- Response: Tim Robbins played the lead role in the movie Forrest Gump.
- Query: Who played the lead role in the movie Forrest Gump?
===================End of Case===================
- Question: {question}
- Response: {response}
- Query:"""

HALLUICATION_INFO_TEMPLATE = """The following response contains hallucinations. Please generate query related to the last sentence of the response, based on both the question and the response, while using the entities from the question as much as possible.
- Question: What is the birthday of the 46th President of the United States?
- Response: Donald John Trump is the 46th President of the United States.
- Query: Who is the 46th President of the United States?
- Question: Which Golden Globe Lifetime Achievement Award did the person who played the protagonist in the movie Forrest Gump win?
- Response: Tim Robbins played the lead role in the movie Forrest Gump.
- Query: Who played the lead role in the movie Forrest Gump.
===================End of Case===================
- Question: {question}
- Response: {response}
- Query:"""


MISSING_INFO_TEMPLATE = """Carefully analyze the response to identify any missing information needed to fully answer the question. Based on the missing information, generate a relevant query.
If **no response** is provided (empty or only whitespace), treat the Known information as empty and list the information required to answer the Question as Missing; then produce a query that requests the missing answer or data.

### Question:
{question}
### Response:
{response}
### Output Format:
```json
{{
    "Question": [Question],
    "Known": [Known Information],
    "Missing": [Missing Information],
    "Query": [Query]
}}
```"""

ENTITY_TEMPLATE = """Please identify all entities in the given sentence. If there are no entities, respond with 'None'. If multiple entities are present, list them in the following format: 'Entity1, Entity2, Entity3'.
- Sentence: {sentence}
- Entities:"""


ENTITY_REPLEACE_TEMPLATE = """Please identify the **entity** in the response that is used to answer the question, and **replace it** with **another entity** of the same type, then return the **modified response**. If no entity is found, modify the response to 'None'.

### Question:
{question}
### Response:
{sentence}
### Output Format:
```json
{{
    "Origial": [Original response],
    "Entity": [Entity],
    "Modified": [Modified response]
}}
```"""

CONFIDENCE_USE_DOCS_SUFFIX = """Above are the documents related to the "Context". """
CONFIDENCE_USE_DOCS_PREFIX = """Below are the documents related to the "Context"."""
CONFIDENCE_USE_DOCS = """When you provide your confidence in "Your Response", please refer to the documents."""
CONFIDENCE_USE_DOCUS_TEMPLATE = "Please provide your confidence in your response based on the above documents.\n"

REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
REFLECTION_AFTER_LAST_TRIAL_HEADER = 'The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
LAST_TRIAL_HEADER = 'You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\n'

TUTOR_ADVICE_HEADER = "I hope you can provide advice to help student as a knowledgeable tutor. I will give you a question, the student's previous reasoning content for the question, and their most recent fail responses."
TUTOR_USE_DOCS = """Below are the documents referred to by student while answering questions."""
TUTOR_USE_DOCS_MIDDLE = """Please provide "Advice" in the same format as the examples based on above documents."""
TUTOR_NOT_USE_DOCS_MIDDLE = """Please provide "Advice" in the same format as the examples."""
TUTOR_ADVICE_MIDDLE = "What's more, please carefully analyze the reason of failed responses and offer constructive advice to help them understand the root causes of these errors and improve their response skills."
ADVICE_TEMPLATE = """{header}

Examples:
{examples}

{docs}{middle}
### Question:
{question}
### Previous reasoning:
{history_resp}
### Fail Response:
{response}
### Output Format:
```json
{{
    "Question": [Question],
    "Reason": [Advice Reason],
    "Advice": [Advice]
}}
```"""

REFLECTION_HEADER = """You’re an advanced reasoning agent capable of self-reflection and continuous improvement. Each problem will provide you with a question, previous excellent responses, and the last failed response."""
REFLECT_USE_DOC = """Below are the documents related to the question."""
REFLECT_USE_DOC_MIDDLE = """Based on the Documents provided above and the Adivce given below, """
REFLECT_NOT_USE_DOC_MIDDLE = """Based on the Adivce given below, """
REFLECTION_MIDDLE = """you can modify the "Fail Response" in the same format as the Examples."""
REFLECTION_TEMPLATE = """{header}

Examples:
{examples}

{docs}{middle}
### Question:
{question}
### Previous reasoning:
{history_resp}
### Fail Response:
{response}
### Advice:
{tutor_ins}
### Output Format:
```json
{{
    "Question": [Question],
    "Reason": [Refine Reason],
    "Modified": [Modified Response]
}}
```"""