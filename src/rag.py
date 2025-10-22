import random
import logging
import numpy as np
import spacy
import torch
from math import exp
from retriever import BM25, SGPT, BGEReranker
from prompts import *
from utils import *
from generators import BasicGenerator, Counter

# GPU_NUMS = torch.cuda.device_count()
GPU_NUMS = 1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BasicRAG:
    def __init__(self, args):
        args = args.__dict__
        for k, v in args.items():
            setattr(self, k, v)
        self.generator = BasicGenerator(self.model_name_or_path, self.__dict__)
        if "retriever" in self.__dict__:
            self.retriever_type = self.retriever
            if self.retriever_type == "BM25":
                # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                self.retriever = BM25(
                    tokenizer = self.generator.tokenizer,
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name,
                    engine = "elasticsearch",
                )
            elif self.retriever_type == "SGPT":
                self.retriever = SGPT(
                    model_name_or_path = self.sgpt_model_name_or_path,
                    sgpt_encode_file_path = self.sgpt_encode_file_path,
                    passage_file = self.passage_file
                )
            elif self.retriever_type == "BGEReranker":
                self.retriever = BGEReranker(
                    model_name_or_path = self.bge_model_name_or_path,
                    index_name = "wiki" if "es_index_name" not in args else self.es_index_name,
                )
            else:
                raise NotImplementedError

        self.counter = Counter()

    def retrieve(self, query, topk=1, max_query_length=64):
        self.counter.retrieve += 1
        if self.retriever_type == "BM25":
            _docs_ids, _doc_titles, docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
                max_query_length = max_query_length,
            )
            separator = np.array([[' | '] * len(docs[0])])
            result = np.char.add(np.char.add(_doc_titles, separator), docs)
            return result[0].tolist()
        elif self.retriever_type == "SGPT":
            docs = self.retriever.retrieve(
                queries = [query],
                topk = topk,
            )
            return docs[0]
        elif self.retriever_type == "BGEReranker":
            _docs_ids, _doc_titles, docs = self.retriever.retrieve(
                queries = [query],
                recall_num = 100,
                topk = topk,
            )
            separator = np.array([[' | '] * len(docs[0])])
            result = np.char.add(np.char.add(_doc_titles, separator), docs)
            return result[0]
        else:
            raise NotImplementedError

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""

    def inference(self, question, demo):
        # non-retrieval
        assert self.query_formulation == "direct"
        prompt = get_answer_prompt([], demo=demo, question=question, text="")
        text, _, _, _ = self.generator.generate(
            prompt,
            max_new_tokens=self.max_length,
        )
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class SingleRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def inference(self, question, demo):
        assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        # 对 topk 个 passage 生成 prompt
        prompt = get_answer_prompt(docs=docs, demo=demo, question=question, text="")
        text, _, _, _ = self.generator.generate(
            prompt,
            self.max_length,
        )
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text


class FixLengthRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def _get_retr_docs_(self, question, ptext):
        if self.query_formulation == "forward_all":
            tmp_all = [question, ptext]
            retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
        elif self.query_formulation == "last_sentence":
            tmp_all = [question, ptext]
            retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
            retrieve_question = retrieve_question.strip()
            retrieve_question = self.get_last_sentence(retrieve_question)
        else:
            retrieve_question = question

        if len(retrieve_question.split()) < 5:
            retrieve_question = question
        try:
            docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
        except:
            docs = []
        return docs

    def inference(self, question, demo):
        ptext = ""
        ptexts = []
        docs = []
        old_len = -1
        while True:
            # 对 topk 个 passage 生成 prompt
            prompt = get_answer_prompt(docs=docs, demo=demo, question=question, text=ptext)
            text, answer, _, _ = self.generator.generate(
                prompt,
                self.generate_length,
                process_gen_text=True,
            )
            if self.use_counter == True:
                self.counter.add_generate(text, self.generator.tokenizer)
            if self.method == "fix-sentence-retrieval":
                # fix sentence
                sentences = list(nlp(answer).sents)
                sentences = [str(sent).strip() for sent in sentences]
                answer = sentences[0] if len(sentences) > 0 else ""
                ptexts.append(answer)
            if self.method == "random-sentence-retrieval":
                sentences = list(nlp(answer).sents)
                sentences = [str(sent).strip() for sent in sentences]
                first_n_sents = ''
                if len(sentences) > 1:
                    first_n = random.randint(1, len(sentences))
                    first_n_sents = sentences[:first_n]
                answer = ' '.join(first_n_sents)
                ptexts.extend(first_n_sents)
            ptext += (" " + answer.strip())
            ptext = ptext.strip()
            # 判断 token 的个数要少于 max_length
            tokens_count = len(self.generator.tokenizer.encode(ptext))
            if tokens_count >= self.max_length or tokens_count <= old_len or "the answer is" in ptext:
                if len(ptexts)==0 or is_ans_unknown(ptexts):
                    ptext = ' '.join(ptexts[:-1])
                    ptext = ptext.strip()
                    docs = self._get_retr_docs_(question, ptext)
                    prompt = get_answer_prompt(
                        docs,
                        demo,
                        question,
                        text=ptext,
                    )
                    text, new_text, _, _ = self.generator.generate(
                        prompt,
                        max_new_tokens=self.max_length,
                        return_logprobs=False,
                        gen_type='answer',
                        process_gen_text=True,
                    )
                    ptext += (' ' + new_text)
                    ptext = ptext.strip()
                break
            old_len = tokens_count

            docs = self._get_retr_docs_(question, ptext)
        return ptext


class TokenRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)
        self.sentence_solver = 'max'

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        if tokens == []:
            tid = 0
        else:
            tid = 1
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):   # 到第一个回车符或者空格为止
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr]]
            probs = np.array(probs)
            if len(probs) == 0:
                p = 0.
            else:
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > self.hallucination_threshold:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1

        # No hallucination
        return text, None, False

    def inference(self, question, demo):
        # assert self.query_formulation == "direct"
        ptext = ""
        old_len = -1
        while True:
            docs = []
            prompt = get_answer_prompt(
                docs=docs,
                demo=demo,
                question=question,
                text=ptext
            )
            new_text, tokens, logprobs = self.generator.generate(
                prompt,
                self.generate_length,
                return_logprobs=True
            )
            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            ptext_, curr, hallucination = self.modifier(new_text, tokens, logprobs)
            ptext += " " + ptext_.strip()
            if hallucination:
                curr = curr.replace("[xxx]", "")
                if self.query_formulation == "direct":
                    retrieve_question = curr
                elif self.query_formulation == "forward_all":
                    tmp_all = [question, ptext_, curr]
                    retrieve_question = " ".join(s for s in tmp_all if len(s) > 0)
                elif self.query_formulation == "last_sentence":
                    ptext = ptext.strip()
                    txt = ptext if len(ptext) > 0 else question
                    retrieve_question = self.get_last_sentence(txt)
                else:
                    raise NotImplemented

                retrieve_question = retrieve_question.strip()
                if len(retrieve_question.split()) < 5:
                    retrieve_question = question
                try:
                    docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                except:
                    docs = []
                prompt = get_answer_prompt(
                    docs = docs,
                    demo = demo,
                    question = question,
                    text = ptext
                )
                text, new_text, _, _ = self.generator.generate(
                    prompt,
                    self.generate_length,
                    return_logprobs=False,
                    process_gen_text=True,
                )
                if self.use_counter == True:
                    self.counter.add_generate(text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                ptext += (" " + new_text.strip())
            ptext = ptext.strip()
            # 判断 token 的个数要少于 max_length
            tokens_count = len(self.generator.tokenizer.encode(ptext))
            if tokens_count >= self.max_length or tokens_count <= old_len or "the answer is" in ptext:
                break
            old_len = tokens_count
        return ptext


class EntityRAG(TokenRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        entity = []
        for sent in sentences:
            doc = nlp(sent)
            li = [ent.text for ent in doc.ents]
            entity.append(li)

        belonging = [-1] * len(text)
        pos = 0
        for tid, tok in enumerate(tokens):
            apr = text[pos:].find(tok) + pos
            assert apr != -1
            for j in range(pos, apr+len(tok)):
                belonging[j] = tid
            pos = apr + len(tok)

        entity_intv = []
        for sid, sent in enumerate(sentences):
            tmp = []
            pos = text.find(sent)
            for ent in entity[sid]:
                apr = text[pos:].find(ent) + pos
                el = belonging[apr]
                er = belonging[apr + len(ent) - 1]
                tmp.append((el, er))
                pos = apr + len(ent)
            entity_intv.append(tmp)

        entity_prob = []
        for ent_itv_per_sent in entity_intv:
            tmp = []
            for itv in ent_itv_per_sent:
                probs = np.array(logprobs[itv[0]:itv[1]+1])
                p = {
                    "avg": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "first": lambda x: x[0] if len(x) > 0 else 0
                }.get(self.entity_solver, lambda x: 0)(probs)
                tmp.append(p)
            entity_prob.append(tmp)

        for sid in range(len(sentences)):
            if len(entity_prob[sid]) == 0:
                continue
            probs = [1 - exp(v) for v in entity_prob[sid]]
            probs = np.array(probs)
            p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(self.sentence_solver, lambda x: 0)(probs)
            if p > self.hallucination_threshold: # hallucination
                # keep sentences before hallucination
                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # replace all hallucinated entities in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                for prob, ent in zip(probs, entity[sid]):
                    apr = curr[pos:].find(ent) + pos
                    if prob > self.hallucination_threshold:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(ent):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(ent)
                return prev, curr, True
        # No hallucination
        return text, None, False

    def inference(self, question, demo):
        return super().inference(question, demo)


class AttnWeightRAG(BasicRAG):
    def __init__(self, args):
        super().__init__(args)

    def modifier(self, text, tokens, attentions, weight):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        tid = 0
        for sid, sent in enumerate(sentences):
            tl, tr = tid, tid
            if sid == len(sentences) - 1:
                tl, tr = tid, len(tokens)
            else:
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr
            # value = attenion * (-log prob)
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)
            value = [attns[i-tl] * weight[i] * (tr-tl) for i in range(tl, tr)]
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]
            if 1 in thres:
                # hallucinated
                if "check_real_words" in self.__dict__ and self.check_real_words:
                    doc = nlp(sent)
                    real_words = set(token.text for token in doc if token.pos_ in
                        ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])
                    def match(tok):
                        for word in real_words:
                            if word in tok:
                                return True
                        return False
                    for i in range(len(thres)):
                        if not match(tokens[tl+i]):
                            thres[i] = 0

                prev = "" if sid == 0 else " ".join(sentences[:sid-1])
                # curr = " ".join(
                #     [tokens[i] if thres[i] == 0 else "[xxx]" for i in range(len(thres))]
                # )
                return True, prev, tokens[tl:tr], thres
        return False, text, None, None

    def keep_real_words(self, prev_text, curr_tokens, curr_hit):
        curr_text = " ".join(curr_tokens)
        all_text = prev_text + " " + curr_text
        input_ids = self.generator.tokenizer.encode(all_text, return_tensors="pt").to(self.generator.model.device)
        input_length = input_ids.shape[1]
        tokens_tmp = self.generator.tokenizer.convert_ids_to_tokens(input_ids[0])

        atten_tmp = self.generator.model(input_ids, output_attentions=True).attentions[-1][0]
        atten_tmp = atten_tmp.to('cpu')
        # merge tokens
        range_ = []
        for i, t in enumerate(tokens_tmp):
            if i == 0 or t.startswith(self.generator.space_token) or input_ids[0][i] == 13:
                range_.append([i, i])
            else:
                range_[-1][-1] += 1
        tokens = []
        for r in range_:
            tokenseq = "".join(tokens_tmp[r[0]: r[1]+1]).replace(self.generator.space_token, "")
            tokens.append(tokenseq)

        # 获取幻觉词对应的 attention
        tl, tr = 0, len(tokens)
        curr_st = len(tokens) - len(curr_tokens)
        attns = []
        for r in range_:
            att = torch.zeros(atten_tmp.shape[0], input_length)
            for i in range(r[0], r[1] + 1):
                att += atten_tmp[:, i]
            att /= (r[1] - r[0] + 1)
            att = torch.mean(att, dim=0)
            att = att[tl:tr]
            if att.shape[0] > 1:
                att = att / sum(att[1:]).item()
            attns.append(att)

        # 计算每个超过阈值的 token 在前文的 attentions
        forward_attns = torch.zeros(tr - tl)
        hit_cnt = 0
        for i in range(len(curr_hit)):
            if curr_hit[i] == 1:
                forward_attns += attns[curr_st + i]
                hit_cnt += 1
        forward_attns /= hit_cnt
        forward_attns = forward_attns.tolist()

        # 分析词性，保留实词对应的 attns
        doc = nlp(all_text)
        real_words = set(token.text for token in doc if token.pos_ in
                      ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token):
            for word in real_words:
                if word in token:
                    return True
            return False

        real_pairs = []
        for i in range(len(tokens)):
            tok, att = tokens[i], forward_attns[i]
            if match(tok):
                real_pairs.append((att, tok, i))

        if "retrieve_keep_top_k" in self.__dict__:
            top_k = min(self.retrieve_keep_top_k, len(real_pairs))
        elif "retrieve_keep_ratio" in self.__dict__:
            top_k = int(len(real_pairs) * self.retrieve_keep_ratio)

        real_pairs = sorted(real_pairs, key = lambda x:x[0])
        real_pairs = real_pairs[:top_k]
        real_pairs = sorted(real_pairs, key = lambda x:x[2])
        return " ".join([x[1] for x in real_pairs])

    def inference(self, question, demo):
        ptext = ""
        docs = []
        old_len = -1
        while True:
            prompt = get_answer_prompt(
                docs=docs,
                demo=demo,
                question=question,
                text=ptext,
            )
            new_text, tokens, attns, logprobs, entropies = self.generator.generate_attn(
                prompt,
                self.generate_length,
                # self.attention_solver,
                use_entropy = self.method == "dragin",
                use_logprob = self.method == "attn_prob"
            )
            weight = entropies if self.method == "dragin" else [-v for v in logprobs]

            if self.use_counter == True:
                self.counter.add_generate(new_text, self.generator.tokenizer)
            hallucination, ptext_, curr_tokens, curr_hit =  self.modifier(new_text, tokens, attns, weight)

            if not hallucination:
                ptext += (" " + new_text.strip())
                ptext = ptext.strip()
            else:
                ptext += (" " + ptext_.strip())
                ptext = ptext.strip()
                forward_all = [question, ptext]
                forward_all = " ".join(s for s in forward_all if len(s) > 0)

                def fetch_last_n_tokens(text, num, tokenizer=self.generator.tokenizer):
                    tokens = tokenizer.tokenize(text)
                    if num >= len(tokens):
                        return text
                    last_n_tokens = tokens[-num:]
                    last_n_sentence = ' '.join(last_n_tokens)
                    return last_n_sentence

                if self.query_formulation == "current":
                    retrieve_question = " ".join(curr_tokens)

                elif self.query_formulation == "current_wo_wrong":
                    retrieve_question = " ".join(
                        list(curr_tokens[i] if curr_hit[i] == 0 else "" for i in range(len(curr_tokens)))
                    )

                elif self.query_formulation == "forward_all":
                    retrieve_question = forward_all

                elif self.query_formulation == "last_sentence":
                    retrieve_question = self.get_last_sentence(forward_all)
                    retrieve_question = retrieve_question.strip()

                elif self.query_formulation == "last_n_tokens":
                    assert "retrieve_keep_top_k" in self.__dict__
                    retrieve_question = fetch_last_n_tokens(
                        forward_all, self.retrieve_keep_top_k)

                elif self.query_formulation == "real_words":
                    retrieve_question = self.keep_real_words(
                        prev_text = question + " " + ptext,
                        curr_tokens = curr_tokens,
                        curr_hit = curr_hit,
                    )
                else:
                    raise NotImplemented

                if len(retrieve_question.split()) < 5:
                    retrieve_question = question
                try:
                    docs = self.retrieve(retrieve_question, topk=self.retrieve_topk)
                except:
                    docs = []
                prompt = get_answer_prompt(
                    docs=docs,
                    demo=demo,
                    question=question,
                    text=ptext,
                )
                new_text, _, _, _ = self.generator.generate(
                    prompt,
                    max_new_tokens = self.generate_length,
                    process_gen_text = False,
                )
                if self.use_counter == True:
                    self.counter.add_generate(new_text, self.generator.tokenizer)
                    self.counter.hallucinated += 1
                new_text = self.get_top_sentence(new_text)
                ptext += (" " + new_text.strip())

            ptext = ptext.strip()
            tokens_count = len(self.generator.tokenizer.encode(ptext))
            if tokens_count >= self.max_length or tokens_count <= old_len or "the answer is" in ptext:
                break
            old_len = tokens_count
        return ptext

