import re
from typing import List, Dict, Any, Union, Optional
import json
import logging
import spacy
nlp = spacy.load("en_core_web_sm")
from prompts import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANSWER_NEW_TOKEN_NUM = 2048

def _extract_first_json_like(text: str) -> Optional[Any]:
    """
    Extract and parse the first balanced JSON object or array found in text.

    Handles nested braces/brackets and string escaping. Returns a parsed object
    (dict or list) if successful; otherwise None.
    """
    # Try object then array
    for opener, closer in (("{", "}"), ("[", "]")):
        for m in re.finditer(re.escape(opener), text):
            start = m.start()
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i+1].strip()
                            try:
                                return json.loads(candidate)
                            except Exception:
                                break  # stop this candidate and continue
            # continue trying next opener occurrence
    return None

def _extract_field_value(raw_text: str, field: str) -> Optional[str]:
    """Extract a single string field value from raw text using regex fallbacks.

    Attempts in order:
    1) Quoted value with double quotes:  "Field": "..."
    2) Quoted value with single quotes:  'Field' = '...'
    3) Unquoted until end-of-line:      Field: ...\n
    Returns the extracted string (without surrounding quotes) or None if not found.
    """
    pattern = rf'''
        \s*['"]?\b{re.escape(field)}\b['"]?\s*[:=]\s*
        (?:
            ['"]((?:[^'"\\]|\\.)*)['"]
        |
            ([^\n\r]+)
        )
    '''
    m = re.search(pattern, raw_text, re.VERBOSE | re.IGNORECASE | re.DOTALL)
    if m:
        if m.group(1) is not None:
            return m.group(1).strip()
        elif m.group(2) is not None:
            return m.group(2).strip()
    return None

def clean_json_txt(json_txt: str) -> Union[Dict[str, Any], str]:
    """
    从可能包含 ```json ... ``` 或 ``` ... ``` 的文本中提取 JSON 并解析为 dict。

    优先级：
      1) 第一个标记为 ```json 的代码块（忽略大小写）
      2) 第一个任意 ``` ... ``` 代码块
      3) 整个输入字符串

    解析失败时返回原始字符串。
    """
    if not isinstance(json_txt, str):
        raise TypeError("json_txt must be a str")

    # 1) 优先查找 ```json ... ```（不区分大小写）
    m = re.search(r'```(?:\s*json\b)[\r\n]*([\s\S]*?)```', json_txt, re.IGNORECASE)
    # 2) 若没有带 json 标记的，再查找任意 ``` ... ``` 代码块
    if not m:
        m = re.search(r'```[\r\n]*([\s\S]*?)```', json_txt)

    if m:
        payload = m.group(1).strip()
    else:
        payload = json_txt.strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        extracted = _extract_first_json_like(json_txt)
        if extracted is not None:
            return extracted
        return json_txt


def find_element_index(lst, element):
    try:
        index = lst.index(element)
        return index
    except ValueError:
        return -1

def split_sentences(text: str) -> List[str]:
    sentences = [sent.text.strip() for sent in nlp(text).sents]
    sentences = [sent for sent in sentences if len(sent) > 0]
    results = []
    i = 0
    while i < len(sentences):
        if re.search(r'\d+\.', sentences[i].strip()) and i < len(sentences) - 1:
            results.append(sentences[i] + " " + sentences[i + 1])
            i += 2
        else:
            results.append(sentences[i])
            i += 1
    return results

def is_complete_sentence(sentence):
    # 检查最后一个字符是否是中英文的句号、问号或感叹号
    return sentence.endswith(('。', '？', '！', '.', '?', '!'))

def process_answer_text(raw_text, pre_answer):
    text = clean_json_txt(raw_text)
    if isintance(text, dict) and "continue" in text:
        return text.get('continue')
    ptns = r'(?i).*?\banswer\s*[:：]\s*'
    pattern = re.compile(ptns, re.DOTALL)
    result = re.sub(pattern, '', text)
    pre_answer = re.sub(pattern, '', pre_answer)
    all_texts = split_sentences(result)
    if len(all_texts) > 1:
        last_txt = all_texts[-1]
        all_texts = all_texts if is_complete_sentence(last_txt) else all_texts[:-1]
    not_in_prompt_texts = [text for text in all_texts if text not in pre_answer]
    return ' '.join(not_in_prompt_texts).strip()

def process_confidence_text(raw_text, conf_type='value'):
    data = clean_json_txt(raw_text)
    score = None
    if isinstance(data, dict):
        score = data.get("Confidence", None)
    if score is None:
        m = re.search(r'(?i)"?confidence"?\s*[:=]\s*([-+]?\d*\.?\d+)', raw_text)
        if m:
            try:
                score = float(m.group(1))
            except Exception:
                score = None
    if score is None:
        print(raw_text)
        raise ValueError("Expected a JSON object")

    if isinstance(score, str):
        score = float(score)

    if conf_type == 'level':
        if isinstance(score, list):
            score = score[0]
        if score > 1:
            score = score / 5
    # If value still >1 but within 1..5, normalize defensively
    if isinstance(score, (int, float)) and 1 < score <= 5 and conf_type != 'level':
        score = score / 5

    if not isinstance(score, (int, float)) or not (0 <= score <= 1):
        logger.warning(f"Invalid confidence score: {score}")
        score = 0

    return score

def process_advice_text(raw_text):
    data = clean_json_txt(raw_text)
    if isinstance(data, dict) and "Advice" in data:
        return data.get("Advice", "")
    extracted = _extract_field_value(raw_text, "Advice")
    if extracted is not None:
        return extracted
    raise ValueError("Missing 'Advice' field")

def process_reflect_text(raw_text):
    data = clean_json_txt(raw_text)
    if isinstance(data, dict) and "Modified" in data:
        return data.get("Modified", "")
    extracted = _extract_field_value(raw_text, "Modified")
    if extracted is not None:
        return extracted
    raise ValueError("Missing 'Modified' field")


def process_keywords_text(raw_text):
    text = raw_text
    ptns_choice = [
        r'(?i).*?\bkeywords\s*[:：]\s*',
        r'(?i).*?\bmy keywords are',
        r'(?i).*?\ba keywords are',
    ]
    for ptns in ptns_choice:
        pattern = re.compile(ptns, re.DOTALL)
        text = re.sub(pattern, '', text)
    text = text.replace('\n', ' ')
    return text

def process_retr_info_text(raw_text):
    data = clean_json_txt(raw_text)
    # Primary: structured JSON
    if isinstance(data, dict) and "Query" in data:
        return data.get("Query", "")

    # Regex fallbacks
    extracted = _extract_field_value(raw_text, "Query")
    if extracted is not None:
        return extracted

    raise ValueError("Missing 'Query' field")


def process_entity_turb_text(raw_text):
    data = clean_json_txt(raw_text)
    # Primary: structured JSON
    if isinstance(data, dict) and "Modified" in data:
        return data.get("Modified", "")

    # Regex fallbacks
    extracted = _extract_field_value(raw_text, "Modified")
    if extracted is not None:
        return extracted

    raise ValueError("Missing 'Query' field")


def is_ans_unknown(answers: List[str]) -> bool:
    unknown_values = [
        "not provided",
        "cannot definitively",
        "not explicitly",
        "not applicable",
        "not available",
        "cannot be",
        "unknown",
        "unsure",
        "not sure",
        "There's no",
        "don't know",
        "inconclusive",
        "not found",
        "uncertainty",
        "none",
        "unable",
        "not enough",
        "not specified",
        "not determined",
        "not disclosed",
        "not revealed",
        "not mentioned",
        "not known",
        "not stated",
        "not directly",
        "not sufficient data",
        "not sufficient information",
        "isn't sufficient",
        "isn't enough",
        "does not",
        "not specify",
    ]
    for idx, answer in enumerate(answers):
        if any(re.search(r'(?i).*?\b{}\b.*'.format(value), answer) for value in unknown_values):
            return idx, True
    pattern = r'(?i)the answer is[：:]?$'
    if re.search(pattern, answer):
        return -1, True
    return None, False


def get_docstr(docs):
    doc_str = ''
    if len(docs) > 0:
        doc_str += "Documents:\n"
        for i, doc in enumerate(docs):
            doc_str += f"[{i+1}] {doc}\n"
        doc_str += ('\n')
    return doc_str


def get_answer_prompt(docs: list, demo: list, question: str, text:str):
    doc_str = get_docstr(docs)
    if len(demo) > 0:
        examples = "Examples:\n" + ("".join([d["case"]+"\n" for d in demo]))
        examples += ('\n')
    else:
        examples = ""
    prompt = ANSWER_QUESTION_TEMPLETE.format(
        examples=examples,
        docs=doc_str,
        use_docs=ANSWER_USE_DOCS_TEMPLATE if len(docs) > 0 else '',
        question=question,
        gen_text=text,
    )
    return prompt


def get_conf_prompt(question:str, history_resp:str, response:str, docs:list, conf_type='value'):
    doc_str = get_docstr(docs)
    if len(docs) > 0:
        doc_str = ('\n' + doc_str + CONFIDENCE_USE_DOCS_SUFFIX)
    template = CONFIDENCE_VALUE_TEMPLATE if conf_type == 'value' else CONFIDENCE_LEVEL_TEMPLATE
    conf_prompt = template.format(
        docs=doc_str,
        question=question,
        history_resp=history_resp,
        response=response,
    )
    return conf_prompt


def get_reason_prompt(docs, reason_pth):
    doc_str = get_docstr(docs)
    reason_prompt = STEP_REASON_ANSWER_TEMPLATE.format(
        docs=doc_str,
        reasoning=reason_pth,
    )
    return reason_prompt
