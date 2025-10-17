
from examples import TUTOR_ADVICE_EXAMPLES, REFLECT_EXAMPLES
from prompts import *
from utils import *
from rag import BasicRAG

class SeqConfidenceRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def _get_seq_confs_level_(self, question:str, history_resp:str, response:str, docs:list, conf_type='value'):

        def __conf_level_in_confs__(confs):
            confs = float(confs)
            if confs >= self.reflection_threshold:
                return 'high'
            elif confs >= self.hallucination_threshold:
                return 'mid'
            else:
                return 'low'

        conf_prompt = get_conf_prompt(
            question=question,
            history_resp=history_resp,
            response=response,
            docs=docs,
            conf_type=conf_type,
        )
        text, confs, _, _ = self.generator.generate(
            conf_prompt,
            max_new_tokens=1024,
            gen_type='confidence',
            process_gen_text=True,
            conf_type=conf_type,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        conf_level = __conf_level_in_confs__(confs)
        if "turbulence" not in self.__dict__ or not self.turbulence:
            if conf_level == 'low':
                return conf_level, 'lack knowledge'
            else:
                return conf_level, 'exact' if conf_level == 'high' else 'reflect'
        else:
            # Add perturbation to rejudge the confidence level of the model in response
            if conf_level == 'low':
                return conf_level, 'lack knowledge'
            else:
                turb_resp = ""
                turb_resp, _, _, _ = self.generator.generate(
                    ENTITY_REPLEACE_TEMPLATE.format(question=question, sentence=response),
                    max_new_tokens=1024,
                    return_logprobs=False,
                    process_gen_text=False,
                )
                if turb_resp == '' or 'None' in turb_resp:   # confs is response confs
                    return conf_level, 'exact' if conf_level == 'high' else 'reflect'

                turb_conf_prompt = get_conf_prompt(
                    question=question,
                    history_resp=history_resp,
                    response=turb_resp,
                    docs=docs,
                    conf_type=conf_type,
                )
                text, mod_confs, _, _ = self.generator.generate(
                    turb_conf_prompt,
                    max_new_tokens=self.generate_confidence_length,
                    gen_type='confidence',
                    process_gen_text=True if conf_type == 'value' else False,
                )
                mod_confs = __conf_level_in_confs__(mod_confs if conf_type == 'value' else text)
                if self.use_counter:
                    self.counter.add_generate(text, self.generator.tokenizer)
                if mod_confs in ['high', 'mid']:
                    return 'low', 'hallucination'
                else:
                    return conf_level, 'exact' if conf_level == 'high' else 'reflect'

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
        text, retr_info, _, _ = self.generator.generate(
            retr_info_prompt,
            max_new_tokens=1024,
            gen_type='retr_info',
            process_gen_text=True,
        )
        return retr_info

    def _get_retr_docs_(self, question, hist_resps, cur_step_ptext, cur_ptext_conf_type, retr_type='retr_query'):
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
            max_new_tokens=self.max_length,
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
            max_new_tokens=self.max_length,
            return_logprobs=False,
            gen_type='reflection',
            process_gen_text=True,
        )
        if self.use_counter:
            self.counter.add_generate(text, self.generator.tokenizer)
        return reflect

    def _get_confs_class_(self, question, history_resp, sent, docs):
        confs_type = self.confs_class if "confs_class" in self.__dict__ else 'value'
        conf_level, conf_type = self._get_seq_confs_level_(question, history_resp, sent, docs, confs_type)

        confs_value = 0
        if "use_reflect" not in self.__dict__ or not self.use_reflect:  # no reflect
            if conf_level=='high':
                confs_value = 1
            else:
                confs_value = -1
            return confs_value, conf_type

        if conf_level=='high':
            confs_value = 1
        elif conf_level=='low':
            confs_value = -1
        else:
            confs_value = 0
        return confs_value, conf_type

    def modifier(self, question, ptext, text, docs):
        """
        按模型对新生成的内容判断自信度进行修改。删除置信度不高的文本
        Args:
            confs_class: str, the type of confidence score, 'value' or 'level'
        Returns:
            ptexts_: list of str, sentences that exceed the hallucination threshold for first. If there are none, retain the first sentence.
            pconfs_: list of float, the confidence score for sentence in the ptexts_
            pconf_types_: list of str, the confindence type for sentence in the ptexts_
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

        # Collect all confidence prompts for batch inference
        conf_prompts = []
        for i, sent in enumerate(sentences):
            prev_batch_text = ' '.join(sentences[:i]) if i > 0 else ''
            current_history = (history_resp + (' ' + prev_batch_text if prev_batch_text else '')).strip()
            conf_prompt = get_conf_prompt(
                question=question,
                history_resp=current_history,
                response=sent,
                docs=docs,
                conf_type=conf_type
            )
            conf_prompts.append(conf_prompt)

        conf_results = self.generator.generate(
            conf_prompts,
            max_new_tokens=1024,
            gen_type='confidence',
            process_gen_text=True,
            conf_type=conf_type
        )

        # Process each result
        for i, (sent, conf_result) in enumerate(zip(sentences, conf_results)):
            modify_text = sent
            confs_value = conf_result[1]
            if i > 0:
                history_resp += (' ' + sentences[i-1])

            # Determine confs_value based on score and reflect logic
            if "use_reflect" not in self.__dict__ or not self.use_reflect:
                if confs_value >= self.hallucination_threshold:  # Assuming high confidence
                    score = 1
                else:
                    score = -1
            else:
                if confs_value >= 0.7:  # high
                    score = 1
                elif score < 0.3:  # low
                    score = -1
                else:  # mid
                    score = 0

            if score == 0 and reflect_tag:
                print(f'cur confs:{score}, performed reflect')
                reft_text = self._reflection_(question, history_resp, sent, docs)
                reft_cons, reft_conf_type = self._get_confs_class_(question, history_resp, reft_text, docs)
                if reft_cons >= 0:
                    modify_text = reft_text
                    confs_value = reft_cons
                    conf_type = reft_conf_type
            elif score < 0:
                print(f'cur confs:{confs_value}, performed hallucination')
                hallucination = True
                reflect_tag = False

            ptexts_.append(modify_text)
            pconfs_.append(score)
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
        while True:
            _, new_text = self._generate_([], demo, question, ptext)
            ptexts_, pconfs_, pconf_types_, hallucination = self.modifier(
                question,
                ptext,
                new_text,
                docs=docs,
            )
            if not hallucination:
                ptext += (' ' + (' '.join(ptexts_)))
                ptexts.extend(ptexts_)
            else:
                if len(ptexts_) > 0:
                    ptexts.extend(ptexts_)
                    ptext += (' ' + (' '.join(ptexts_)))
                    pre_seq = ptexts_[-1]
                    pre_seq_conf_type = pconf_types_[-1]
                else:
                    pre_seq = ""
                    pre_seq_conf_type = 'lack knowledge'
                retr_num += 1
                docs, retr_quest = self._get_retr_docs_(question, ptexts, pre_seq, pre_seq_conf_type)
                _, new_text = self._generate_(docs=docs, demo=demo, question=question, ptext=ptext, generate_length=128)
                ptexts.append(new_text)
                ptext += (' ' + new_text)
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
                    docs, _ = self._get_retr_docs_(question, [], unknown_info, 'lack knowledge')
                    _, new_text = self._generate_(docs, demo, question, ptext, generate_length=self.max_length)
                    ptext += (' ' + new_text)
                    ptext = ptext.strip()
                break
            old_len = cur_len

        return ptext
