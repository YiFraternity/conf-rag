import logging
import numpy as np
import spacy
import torch
from typing import Callable, Optional, Dict, Any, Union, List
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from vllm import LLM, SamplingParams
from openai import OpenAI
from prompts import *
from utils import *

GPU_NUMS = max(1, torch.cuda.device_count())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BasicGenerator:

    def __update_generate_config__(self, params):
        for k in self.generate_config.keys():
            if k in params:
                self.generate_config[k] = params[k]
        if self.generate_config["do_sample"] is False:
            self.generate_config["top_k"] = -1
            self.generate_config['temperature'] = 0
            self.generate_config['top_p'] = 1.0

    def __init__(self, model_name_or_path: str, params: Optional[Dict[str, Any]] = None) -> None:
        if params is None:
            params = {}

        logger.info(f"Initializing generator with model: {model_name_or_path}")

        self.generate_config = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "num_beams": 1,
            "do_sample": True,
        }
        self.__update_generate_config__(params)
        use_openai_flag = params.get("use_openai")
        inferred_openai = (
            (isinstance(use_openai_flag, bool) and use_openai_flag) or
            (params.get("openai_api_key") is not None) or
            (params.get("openai_base_url") is not None)
        )
        if inferred_openai:
            self.use_openai = True
            self.model_name = model_name_or_path
            self.client = OpenAI(
                api_key=params.get("openai_api_key"),
                base_url=params.get("openai_base_url")
            )
            # OpenAI(-compatible) models don't need a tokenizer for generation, keep one for compatibility
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.use_openai = False
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

            if 'use_vllm' in params and not params['use_vllm']:
                self.model_config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=True,
                    output_attentions=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map="auto",
                    trust_remote_code=True,
                ).eval()
                if self.model_config.model_type in ["llama", "qwen2"]:
                    self.space_token = "Ġ"  # Llama3为`Ġ`，Llama2为`▁`
                else:
                    self.space_token = self.tokenizer.tokenize(' ')[0]

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.use_vllm = False
            else:
                self.model = LLM(model=model_name_or_path, max_model_len=4096, tensor_parallel_size=GPU_NUMS)
                self.use_vllm = True

    def _get_chat_message_(self, prompt):
        message = [
            {"role": "system", "content": "You are a concise and efficient assistant. Please proceed directly with reasoning and answering, avoiding any unrelated phrases such as 'Let me help,', 'Let’s analyze the information,', or similar expressions. You must response in Enlish and not repeat the reasoning."},
            {"role": "user", "content": prompt}
        ]
        return message

    def generate(
            self,
            input_text: Union[List, str],
            max_new_tokens: int,
            return_logprobs: bool = False,
            gen_type: str = "answer",
            process_gen_text: bool = False,
            **kwargs,
    ):
        """Generate text using the specified model.

        Args:
            input_text: The input prompt or text to generate from (str or List[str])
            max_new_tokens: Maximum number of tokens to generate
            return_logprobs: Whether to return token log probabilities
            gen_type: Type of generation, one of ['answer', 'confidence', 'advice', 'reflection', 'keywords', 'retr_info']
            process_gen_text: Whether to process the generated text based on gen_type

        Returns:
            If input_text is str, returns Union[str, tuple]
            If input_text is List[str], returns List[Union[str, tuple]]
        """
        process_fns = {
            'answer': process_answer_text,
            'confidence': process_confidence_text,
            'advice': process_advice_text,
            'reflection': process_reflect_text,
            'keywords': process_keywords_text,
            'retr_info': process_retr_info_text,
            'entity_turb': process_entity_turb_text,
        }
        if gen_type not in process_fns:
            raise ValueError(f"gen_type {gen_type} is not supported. Must be one of {list(process_fns.keys())}")
        process_text = process_fns[gen_type]
        if gen_type == 'answer':
            kwargs['pre_answer'] = input_text
        # Handle batch input
        if isinstance(input_text, list):
            return self._generate_batch(input_text, max_new_tokens, process_text, process_gen_text, **kwargs)
        return self._generate_single(input_text, max_new_tokens, return_logprobs, process_text, process_gen_text, **kwargs)

    def _generate_batch(
            self,
            input_text: List[str],
            max_new_tokens: int,
            process_text: Callable,
            process_gen_text: bool = False,
            **kwargs,
    ) -> List[tuple[str, Union[str, float], None, None]]:
        messages = [self._get_chat_message_(prompt) for prompt in input_text]
        sampling_params = SamplingParams(
            temperature=self.generate_config['temperature'],
            top_p=self.generate_config['top_p'],
            top_k=self.generate_config['top_k'],
            min_p=0.0 if self.generate_config['temperature']==0 else 1,
            max_tokens=max_new_tokens,
        )
        outputs_t = self.model.chat(
            messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        pred_lst = []
        for o_t in outputs_t:
            text = o_t.outputs[0].text
            if process_gen_text:
                processed_text = process_text(text, **kwargs)
            else:
                processed_text = text
            pred_lst.append((text, processed_text, None, None))
        return pred_lst

    def _generate_single(
            self,
            input_text: str,
            max_new_tokens: int,
            return_logprobs: bool,
            process_text: Callable,
            process_gen_text: bool,
            **kwargs,
    ) -> tuple[str, Union[str, float], None, None]:
        """Generate text using the specified model.

        Args:
            input_text: The input prompt or text to generate from
            max_new_tokens: Maximum number of tokens to generate
            return_logprobs: Whether to return token log probabilities
            gen_type: Type of generation, one of ['answer', 'confidence', 'advice', 'reflection', 'keywords', 'retr_info']
            process_gen_text: Whether to process the generated text based on gen_type

        Returns:
            If return_logprobs is True, returns a tuple of (text, tokens, logprobs)
            Otherwise, returns the generated text as a string
        """
        if hasattr(self, 'use_openai') and self.use_openai:
            messages = self._get_chat_message_(input_text)

            # Prepare generation parameters
            params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": min(max_new_tokens, self.generate_config.get("max_tokens", 1024)),
                "temperature": self.generate_config.get("temperature", 0.7),
                "top_p": self.generate_config.get("top_p", 1.0),
            }

            # Add optional parameters if specified
            if "frequency_penalty" in self.generate_config:
                params["frequency_penalty"] = self.generate_config["frequency_penalty"]
            if "presence_penalty" in self.generate_config:
                params["presence_penalty"] = self.generate_config["presence_penalty"]
            if "stop" in self.generate_config:
                params["stop"] = self.generate_config["stop"]

            if return_logprobs:
                params["logprobs"] = True
                params["top_logprobs"] = 5  # Number of top logprobs to return

            try:
                response = self.client.chat.completions.create(**params)

                if return_logprobs:
                    # For OpenAI, we don't get token-by-token logprobs in streaming mode,
                    # so we'll just return the full text and tokenize it
                    text = response.choices[0].message.content
                    tokens = self.tokenizer.tokenize(text)
                    # Dummy logprobs since OpenAI doesn't provide token-level logprobs in chat completions
                    logprobs = [0.0] * len(tokens)
                    return text, tokens, logprobs
                else:
                    text = response.choices[0].message.content
                    if process_gen_text:
                        text = process_text(text)
                    return text

            except Exception as e:
                logger.error(f"Error in OpenAI API call: {str(e)}")
                raise

        # Handle local model generation
        message = self._get_chat_message_(input_text)
        if not self.use_vllm:
            input_text_formatted = self.tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer.encode(input_text_formatted, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)
            input_length = input_ids.shape[1]
            attention_mask = torch.ones_like(input_ids)

            if return_logprobs:
                outputs = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = max_new_tokens,
                    return_dict_in_generate = True,
                    output_scores = True,
                    **self.generate_config,
                )
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )

                generated_tokens = outputs.sequences[:, input_length:]
                text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True) # text = "".join(tokens)
                tokens = [self.tokenizer.decode(t, skip_special_tokens=True) for t in generated_tokens[0]]
                special_tokens_index = [idx for idx, t in enumerate(tokens) if t == '']
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]
                assert len(tokens) == len(logprobs)
                tokens = [ t for idx, t in enumerate(tokens) if idx not in special_tokens_index]     # remove sepical token
                logprobs = [p for idx, p in enumerate(logprobs) if idx not in special_tokens_index]     # remove sepical token
                return text, tokens, logprobs

            else:
                outputs = self.model.generate(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    max_new_tokens = max_new_tokens,
                    **self.generate_config,
                )
                generated_tokens = outputs[:, input_length:]
                text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        else:
            sampling_params = SamplingParams(
                temperature=self.generate_config['temperature'],
                top_p=self.generate_config['top_p'],
                top_k=self.generate_config['top_k'],
                min_p=0.0 if self.generate_config['temperature']==0 else 1,
                max_tokens=max_new_tokens,
            )
            outputs = self.model.chat(
                message,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            text = outputs[0].outputs[0].text
        if process_gen_text:
            processed_text = process_text(text, **kwargs)
        else:
            processed_text = text
        return text, processed_text, None, None

    def generate_attn(
            self,
            input_text,
            max_new_tokens,
            solver="max",
            use_entropy = False,
            use_logprob = False,
        ):
        message = self._get_chat_message_(input_text)
        input_text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            return_dict_in_generate = True,
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True, return_dict=True).attentions[-1][0]
        if solver == "max":
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]

        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(' ', "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                tmp.append(v.cpu())
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq)
        else:
            seqentropies = None

        return text, seqlist, attns, seqlogprobs, seqentropies

class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0
        self.reflect = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve,
            "reflect_count": self.reflect - other_counter.reflect,
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated,
            "token_count": self.token - other_counter.token,
            "sentence_count": self.sentence - other_counter.sentence
        }
