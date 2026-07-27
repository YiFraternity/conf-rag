
from examples import TUTOR_ADVICE_EXAMPLES, REFLECT_EXAMPLES
from prompts import *
from utils import *
from utils import _coerce_text_value
from rag import BasicRAG

class SeqConfidenceRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def _get_experiment_condition(self):
        condition = getattr(self, "experiment_condition", None)
        if condition in (None, ""):
            return None
        valid_conditions = {"baseline", "low_only", "mid_only", "high_only"}
        if condition not in valid_conditions:
            raise ValueError(f"Unsupported experiment_condition: {condition}")
        return condition

    def _should_retrieve_for_level(self, level):
        condition = self._get_experiment_condition()
        if condition == "baseline":
            return False
        if condition == "low_only":
            return level == -1
        if condition == "mid_only":
            return level == 0
        if condition == "high_only":
            return level == 1
        return level < 0

    def _should_reflect_for_level(self, level):
        condition = self._get_experiment_condition()
        if condition is not None:
            return False
        return level == 0

    def _build_trace_event(self, loop_id, sentence_id, sentence, conf_value, conf_type, docs):
        return {
            "loop_id": loop_id,
            "sentence_id": sentence_id,
            "sentence": sentence,
            "confidence_value": conf_value,
            "confidence_type": conf_type,
            "trigger_retrieval": False,
            "retrieval_disabled": False,
            "retrieval_query": "",
            "retrieved_docs_count": len(docs) if docs is not None else 0,
            "post_retrieval_text": "",
        }

    def _get_seq_confs_value_(self, question:str, history_resp:str, sentences:List[str], docs:list, conf_type='value') -> List[float]:
        conf_prompts = []
        for i, sent in enumerate(sentences):
            prev_batch_text = ' '.join(sentences[:i]) if i > 0 else ''
            current_history = (history_resp + (' ' + prev_batch_text if prev_batch_text else '')).strip()
            conf_prompt = get_conf_prompt(
                question=question,
                history_resp=current_history,
                response=sent,
                docs=docs,
                conf_type=conf_type,
            )
            conf_prompts.append(conf_prompt)

        conf_results = self.generator.generate(
            conf_prompts,
            max_new_tokens=1024,
            gen_type='confidence',
            process_gen_text=True,
            conf_type=conf_type
        )

        confidence_value_lst = []
        for conf_result in conf_results:
            confidence_value_lst.append(conf_result[1])
            text = conf_result[0]
            if self.use_counter:
                self.counter.add_generate(text, self.generator.tokenizer)
        assert len(confidence_value_lst) == len(sentences)
        return confidence_value_lst

    def _get_confs_(self, question:str, history_resp:str, sentences:List[str], docs:list, conf_type='value') -> List[int]:
        raw_seqs_conf_vals = self._get_seq_confs_value_(
            question=question,
            history_resp=history_resp,
            sentences=sentences,
            docs=docs,
            conf_type=conf_type,
        )

        if "turbulence" in self.__dict__ and self.turbulence:
            turb_prompts = []
            for sen in sentences:
                prompt = ENTITY_REPLEACE_TEMPLATE.format(question=question, sentence=sen)
                turb_prompts.append(prompt)

            turb_resps = self.generator.generate(
                turb_prompts,
                max_new_tokens=1024,
                gen_type='entity_turb',
                process_gen_text=True,
            )

            turb_resp_texts = [_[0] for _ in turb_resps]
            turb_resps_conf_vals = self._get_seq_confs_value_(
                question=question,
                history_resp=history_resp,
                sentences=turb_resp_texts,
                docs=docs,
                conf_type=conf_type,
            )
            raw_seqs_conf_vals = [raw - turb for raw, turb in zip(raw_seqs_conf_vals, turb_resps_conf_vals)]

        levels = []
        for val in raw_seqs_conf_vals:
            if val >= self.reflection_threshold:
                levels.append(1)
            elif val >= self.hallucination_threshold:
                levels.append(0)
            else:
                levels.append(-1)
        return levels

    def _generate_(self, docs=[], demo=[], question='', ptext='', qtype='answer', generate_length=-1):
        if qtype == 'reason':
            prompt = get_reason_prompt(docs, reason_pth=question)
        else:
            prompt = get_answer_prompt(docs, demo, question, ptext)
        # 当前轮次的新文本
        text, new_text, _, _ = self.generator.generate(
            prompt,
            max_new_tokens=self.generate_length if generate_length==-1 else generate_length,
            return_logprobs=False,
            gen_type='answer',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, new_text

    def _get_keywords_(self, addition_info):
        if addition_info == "":
            return ""
        keyword_prompt = KEYWORDS_TEMPLATE.format(sentence=addition_info)
        _, keywords, _, _ = self.generator.generate(
            keyword_prompt,
            max_new_tokens=10,
            gen_type='keywords',
            process_gen_text=True,
        )
        return keywords

    def _get_retr_query_(self, question, response):
        response = response.strip()
        if len(response) == 0:
            response = ""
        retr_info_prompt = MISSING_INFO_TEMPLATE.format(
            question=question,
            response=response,
        )
        _, retr_info, _, _ = self.generator.generate(
            retr_info_prompt,
            max_new_tokens=1024,
            gen_type='retr_info',
            process_gen_text=True,
        )
        return retr_info

    def _get_retr_docs_(self, question, hist_resps, cur_step_ptext, retr_type='retr_query'):
        """
        Args:
            question: str
            historys: list of str
            cur_step_ptext: str
            retr_type: str [retr_query, retr_query_keywords]
        """
        history = " ".join(hist_resps).strip()
        if self.query_formulation == "direct":
            retrieve_question = question
        elif self.query_formulation == "forward_all":
            forward_all = [question, cur_step_ptext]
            forward_all = " ".join(s for s in forward_all if len(s) > 0)
            forward_all = forward_all.replace("[xxx].", "")
            retrieve_question = forward_all
        elif self.query_formulation == "last_sentence":
            forward_all = [question, history]
            forward_all = " ".join(s for s in forward_all if len(s) > 0)
            retrieve_question = forward_all
            retrieve_question = self.get_last_sentence(forward_all)
        elif self.query_formulation == "query_and_last_sentence":
            forward_all = [history]
            forward_all = " ".join(s for s in forward_all if len(s) > 0)
            retrieve_question = self.get_last_sentence(forward_all)
            retrieve_question = question + " " + retrieve_question
        elif self.query_formulation == "generate_query":
            if retr_type == "retr_query":
                retrieve_question = self._get_retr_query_(question, history)
            elif retr_type == "retr_query_keywords":
                keywords = self._get_keywords_(cur_step_ptext)
                retrieve_question = question + " " + keywords
        else:
            raise NotImplemented

        retrieve_question = retrieve_question.strip()
        if len(retrieve_question.split()) < 5:
            retrieve_question = question
        try:
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
        except Exception as e:
            docs = []
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"retrieve failed: {e}")
            else:
                print(f"retrieve failed: {e}")
        return docs, retrieve_question

    def _reflection_(self, question, history_resp, response, docs=[]):
        """
        # 反思需要两次生成
        # 1. tutor-advice: 用于指导从哪个层面思考
        # 2. Refine: 用于提升回复的质量
        """
        if self.use_counter:
            self.counter.reflect += 1
        doc_str = get_docstr(docs)
        tutor_data = {
            "header": TUTOR_ADVICE_HEADER,
            "examples": TUTOR_ADVICE_EXAMPLES,
            "docs": (TUTOR_USE_DOCS + '\n' + doc_str) if len(docs) > 0 else doc_str,
            "middle": (TUTOR_USE_DOCS_MIDDLE if len(docs) > 0 else TUTOR_NOT_USE_DOCS_MIDDLE) +  " " + TUTOR_ADVICE_MIDDLE,
            "question": question,
            "history_resp": history_resp.replace('\n', ' '),
            "response": response,
        }
        advice_prompt = ADVICE_TEMPLATE.format(**tutor_data)
        text, advice, _, _ = self.generator.generate(
            advice_prompt,
            max_new_tokens=1024,
            return_logprobs=False,
            gen_type='advice',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)

        reft_prompt = {
            "header": REFLECTION_HEADER,
            "examples": REFLECT_EXAMPLES,
            "docs": (REFLECT_USE_DOC + '\n' + doc_str + '\n') if len(docs) > 0 else doc_str,
            "middle": (REFLECT_USE_DOC_MIDDLE if len(docs) > 0 else REFLECT_NOT_USE_DOC_MIDDLE) + REFLECTION_MIDDLE,
            "question": question,
            "history_resp": history_resp.replace('\n', ' '),
            "response": response,
            "tutor_ins": advice,
        }
        reflect_prompt = REFLECTION_TEMPLATE.format(**reft_prompt)
        text, reflect, _, _ = self.generator.generate(
            reflect_prompt,
            max_new_tokens=1024,
            return_logprobs=False,
            gen_type='reflection',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return reflect

    def modifier(self, question, ptext, text, docs):
        """
        按模型对新生成的内容判断自信度进行修改。删除置信度不高的文本
        Args:
            confs_class: str, the type of confidence score, 'value' or 'level'
        Returns:
            ptexts_: list of str, sentences that exceed the hallucination threshold for first. If there are none, retain the first sentence.
            pconfs_: list of float, the confidence score for sentence in the ptexts_
            hallucination: bool, whether the text is hallucinated
        """
        hallucination = False
        reflect_tag = True
        sentences = split_sentences(text)
        if len(sentences) == 0:
            return [], [], [], hallucination

        ptexts_, pconfs_, pconf_types_ = [], [], []
        history_resp = ptext
        conf_type=self.confs_class if "confs_class" in self.__dict__ else 'value'

        conf_levels = self._get_confs_(
            question=question,
            history_resp=history_resp,
            sentences=sentences,
            docs=docs,
            conf_type=conf_type,
        )

        for sent, level in zip(sentences, conf_levels):
            modify_text = _coerce_text_value(sent)

            if self._should_reflect_for_level(level) and reflect_tag:
                print(f'cur confs:{level}, performed reflect')
                modify_text = _coerce_text_value(self._reflection_(question, history_resp, sent, docs))

            elif self._should_retrieve_for_level(level):
                print(f'cur confs:{level}, performed hallucination')
                hallucination = True
                reflect_tag = False

            ptexts_.append(modify_text)
            pconfs_.append(level)
            pconf_types_.append(conf_type)

        assert len(sentences) == len(pconfs_)

        hall_index = find_element_index(pconfs_, -1)
        if hall_index >= 0:
            ptexts_ = ptexts_[:hall_index]
            pconfs_ = pconfs_[:hall_index]
            pconf_types_ = pconf_types_[:hall_index]
        return ptexts_, pconfs_, pconf_types_, hallucination

    def inference(self, question, demo):
        ptext = ""     # 用于存储置信度高的序列，以及后续不可提升序列置信度的句子
        ptexts = []
        docs = []
        old_len = -1
        retr_num = 0
        loop_id = 0
        trace_events = []
        while True:
            _, new_text = self._generate_([], demo, question, ptext)
            ptexts_, pconfs_, pconf_types_, hallucination = self.modifier(
                question,
                ptext,
                new_text,
                docs=docs,
            )
            ptexts_ = [_coerce_text_value(text) for text in ptexts_]
            curr_events = []
            for sent_idx, (sent, conf_value) in enumerate(zip(ptexts_, pconfs_)):
                curr_events.append(
                    self._build_trace_event(
                        loop_id,
                        sent_idx,
                        sent,
                        conf_value,
                        pconf_types_[sent_idx] if sent_idx < len(pconf_types_) else "",
                        docs,
                    )
                )
            if not hallucination:
                ptext += (' ' + (' '.join(ptexts_)))
                ptexts.extend(ptexts_)
                trace_events.extend(curr_events)
            else:
                if len(ptexts_) > 0:
                    ptexts.extend(ptexts_)
                    ptext += (' ' + (' '.join(ptexts_)))
                    pre_seq = ptexts_[-1]
                else:
                    pre_seq = ""
                if curr_events:
                    trace_events.extend(curr_events[:-1])
                    curr_events[-1]["trigger_retrieval"] = not getattr(self, "disable_retrieval", False)
                    curr_events[-1]["retrieval_disabled"] = getattr(self, "disable_retrieval", False)
                if getattr(self, "disable_retrieval", False):
                    if curr_events:
                        trace_events.append(curr_events[-1])
                    docs = []
                else:
                    retr_num += 1
                    docs, retr_quest = self._get_retr_docs_(question, ptexts, pre_seq)
                    if curr_events:
                        curr_events[-1]["retrieval_query"] = retr_quest
                        curr_events[-1]["retrieved_docs_count"] = len(docs)
                    _, new_text = self._generate_(docs=docs, demo=demo, question=question, ptext=ptext, generate_length=128)
                    new_text = _coerce_text_value(new_text)
                    ptexts.append(new_text)
                    ptext += (' ' + new_text)
                    if curr_events:
                        curr_events[-1]["post_retrieval_text"] = new_text
                        trace_events.append(curr_events[-1])
            ptext = ptext.strip()
            cur_len = len(self.generator.tokenizer.encode(ptext)) if ptext != "" else 0

            if "the answer is" in ptext or \
                    cur_len >= self.max_length or \
                    cur_len <= old_len or \
                    retr_num >= self.max_retrieve:
                idx, unknown = is_ans_unknown(ptexts)
                if len(ptexts)==0 or unknown:
                    ptext = ' '.join(ptexts[:idx])
                    # ptext = ''
                    unknown_info = ptexts[idx] if idx and idx >= 0 else ptext
                    unknown_info = unknown_info.strip() if len(unknown_info)>0 else ""
                    docs, _ = self._get_retr_docs_(question, [], unknown_info)
                    _, new_text = self._generate_(docs, demo, question, ptext, generate_length=self.max_length)
                    ptext += (' ' + new_text)
                    ptext = ptext.strip()
                break
            old_len = cur_len
            loop_id += 1

        if getattr(self, "save_trace", False):
            return ptext, {
                "experiment_condition": self._get_experiment_condition(),
                "retrieval_enabled": not getattr(self, "disable_retrieval", False),
                "events": trace_events,
            }
        return ptext
