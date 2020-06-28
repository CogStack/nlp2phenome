import spacy
import utils
import pandas as pd
from os import listdir
from os.path import isfile, join, split


class AbstractedSentence(object):
    def __init__(self, seq):
        self._seq = 0
        self._abstracted_tokens = []
        self._text = None
        self._parsed = None

    @property
    def seq(self):
        return self._seq

    @seq.setter
    def seq(self, value):
        self._seq = value

    def add_token(self, t):
        self._abstracted_tokens.append(t)

    @property
    def tokens(self):
        return self._abstracted_tokens

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value

    def get_parsed_tree(self, nlp):
        """
        use spacy instance to parse the sentence
        :param nlp: a spacy instance
        :return: dependency tree
        """
        if self._parsed is not None:
            return self._parsed
        if self.text is None:
            return None
        self._parsed = nlp(self.text)
        return self._parsed

    def locate_pos(self, str):
        return self._text.find(str)

    def get_abstaction_by_pos(self, pos, nlp):
        doc = self.get_parsed_tree(nlp)
        token = None
        if doc is not None:
            for t in doc:
                if t.idx + len(t.text) == pos:
                    token = t
        if token is not None:
            ta = TokenAbstraction(token, doc)
        else:
            return None
        return ta
    
    def get_related_tokens(self, t):
        ret = []
        for tk in self._parsed:
            if tk.head == t:
                ret.append(tk)
            print(tk.text, tk.dep_, tk.head)
        return ret


class TokenAbstraction(object):
    def __init__(self, token, doc):
        self._t = token
        self._d = doc
        self._children = []
        self._root = None
        self._subject = None
        self._verbs = None
        self._vcontext = []
        self.do_abstract()

    @property
    def vcontext(self):
        return self._vcontext
        
    @property
    def children(self):
        return self._children

    @property
    def root(self):
        return self._root

    @property
    def subject(self):
        return self._subject

    @property
    def verbs(self):
        return self._verbs
    
    @property
    def token(self):
        return self._t

    def do_abstract(self):
        self._children = [t for t in self._t.children]
        t = self._t
        r = t
        while (t.head != t) and t.dep_ not in ['ROOT', 'relcl', 'acl', 'advcl']:
            t = t.head
            if t.dep_ in ['ccomp'] :
                self._subject = [s for s in t.children if s.dep_ in [u"nsubj", 'nsubjpass', 'ROOT', 'pobj']]
            if t.pos_ in ['VERB'] :
                self._vcontext += [s for s in t.children if s.dep_ in ["neg", 'advmod']]
            r = t
        if t is not None:
            self._verbs = [v for v in t.children if v.pos_ == u"VERB"]
            if t.dep_ in ['relcl', 'acl']:
                self._subject = [t.head]
            else:
                if len(self._vcontext) == 0:
                    self._vcontext += [s for s in t.children if s.dep_ in ["neg", 'advmod']]
                if self._subject is None:
                    self._subject = [s for s in t.children if s.dep_ in [u"nsubj", 'nsubjpass', 'ROOT']]
        self._root = r
        
    def find_parent_dep_until_(self, t, pos):
        while (t.head != t) and t.dep_ != pos:
            t = t.head
        if t.dep_ != pos:
            return None
        else:
            return t
    
    def do_abstract_waterfall(self, entity_start, entity_end):
        t = self._t
        seq = []
        while (t.head != t) and t.dep_ not in ['ROOT', 'relcl', 'acl', 'advcl']:
            t = t.head            
            if t.idx > entity_end or (t.idx + len(t.text) < entity_start):
                seq.append((t.text, t.dep_, t.pos_) )
        seq.reverse()
        return seq
    
    def do_abstract_descendent(self):
        return [c for c in self._t.children]


    def to_dict(self):
        return {'children': [t.text for t in self.children], 'root': self.root.text, 'subject': [s.text for s in self.subject], 'verbs': [v.text for v in self.verbs]}



class MentionPattern(object):
    def __init__(self, pattern_folder, cui2icd, csv_file, ann_folder):
        self._ptn_folder = pattern_folder
        self._ref_good_ptns = None
        self._ref_bad_ptns = None
        self._csv_file = csv_file
        self._cui2icd = cui2icd
        self._df = None
        self._nlp = get_nlp_lg()
        self._ann_folder = ann_folder
        self.load()

    def load(self):        
        self._df = pd.read_csv(self._csv_file)

    @staticmethod
    def load_ref_patterns(ptn_folder, ignore_chapter):
        good_p = MentionPattern.load_patterns(ptn_folder, to_load=lambda f: f.find('good') >0 and f.find('%s_' % ignore_chapter)!=0)
        bad_p = MentionPattern.load_patterns(ptn_folder, to_load=lambda f: f.find('bad') >0 and f.find('%s_' % ignore_chapter)!=0)
        return good_p, bad_p

    @staticmethod
    def get_sent_by_pos(sents, s, e):
        for sent in sents:
            if sent['start'] <= s and sent['end'] >= e:
                return sent
        return None

    def read_semehr_anns(self, doc_anns, container):
        """
        doc_anns - [{'d': fk, 'ann': a, 'label': self.label}]
        """
        cur_d = None
        cur_sents = None
        for da in doc_anns:
            d = 'se_ann_%s.json' % da['d']
            if d != cur_d:
                cur_sents = utils.load_json_data(join(self._ann_folder, d))['sentences']
                cur_d = d
            a = da['ann']
            ch = self._cui2icd[a.cui]
            sent = MentionPattern.get_sent_by_pos(cur_sents, a.start, a.end)
            win = self._df[self._df['doc_id'] == da['d']]['text'].iloc[0][sent['start']:sent['end']]
            container.append({'ch': ch, 'd': da['d'], 's': a.start, 'e': a.end, 's_s': sent['start'], 's_e': sent['end'], 'win':win})

    def abstract_ann_pattern(self, ann):
        abss = AbstractedSentence(2)
        abss.text = ann['win']
        result = abss.get_abstaction_by_pos(ann['e'] - ann['s_s'], self._nlp)
        if result is not None:
            # abss.get_related_tokens(result.token)
            ptn = result.do_abstract_waterfall(ann['s'] - ann['s_s'], ann['e'] - ann['s_s'])
            return {'pattern': ptn, "subject": result.subject, "vcontect": result.vcontext}
        else:
            return None

    def classify_anns(self, anns):
        preds = []
        for ann in anns:
            ret = self.abstract_ann_pattern(ann)
            if ret is not None:
                good_ref, bad_ref = MentionPattern.load_ref_patterns(self._ptn_folder, ann['ch'])
                good_match = MentionPattern.compute_similar_from_ref(ret, good_ref, self._nlp)
                bad_match = MentionPattern.compute_similar_from_ref(ret, bad_ref, self._nlp)
                ctx = '|'.join([e[0] for e in ret['pattern']])
                cls = MentionPattern.classify_by_pattern_matches(good_match, bad_match, self._nlp)

                print('>>>', ctx, good_match, bad_match, cls)
                preds.append(cls)
            else:
            	preds.append(-1)
        return preds

    def predict(self, doc_anns):
        anns = []
        self.read_semehr_anns(doc_anns, anns)
        return self.classify_anns(anns)

    @staticmethod
    def load_patterns(ptn_folder, to_load=lambda f: True):
        return [utils.load_json_data(join(ptn_folder, f)) for f in listdir(ptn_folder) if to_load(f) and isfile(join(ptn_folder, f))]

    @staticmethod
    def sim_seqs(s1, s2, nlp, last_k=2):
        scores = 0.0
        k = min(last_k, len(s1), len(s2))
        for i in range(1, k+1):
            t1, t2 = nlp(' '.join([s1[-1 * i], s2[-1 * i]]))
            if t1.vector_norm > 0 and t2.vector_norm > 0:
                scores += t1.similarity(t2)
        return scores / k

    @staticmethod
    def get_pattern_group(p):    
        mp = p if len(p)<=2 else p[-2:]
        return '-'.join([e[2] for e in mp])

    @staticmethod
    def compute_similar_from_ref(ret, ref_good_ptns, nlp, threshold=0.7):
        p = ret['pattern']
        ctxt = '|'.join([e[0] for e in p])
    #     print('>>>working on %s' % ctxt)
        if len(ctxt) == 0:
            return None
        grp = MentionPattern.get_pattern_group(p)
        entried_scores = []
        for ref_ptn in ref_good_ptns:
            if grp in ref_ptn:            
                for inst in ref_ptn[grp]:
                    score = MentionPattern.sim_seqs([e[0] for e in p], ref_ptn[grp][inst]['list'], nlp)
                    if score > threshold:
                        entried_scores.append((score, ref_ptn[grp][inst]['freq']))
        #                 print('\tvs %s: score %s, %s' % (inst, score, ref_good_ptns[grp][inst]['freq']))
        if len(entried_scores) > 0:
            total = sum([s[0] * s[1] for s in entried_scores])
            supports = sum([s[1] for s in entried_scores])
            avg_score = total / supports
            # print('\tscore %s, support %s, %s|%s' % (avg_score, supports, ret['subject'], ret['vcontect']))
            return {'score': avg_score, 'supports': supports, 'subject': [t.text for t in ret['subject']], 
                    'context': [t.text for t in ret['vcontect']]}
        else:
            return None
                
    @staticmethod        
    def classify_by_pattern_matches(good_match, bad_match, nlp, 
                                    bad_subjs=['son', 'daughter', 'manager', 'wife', 'I', 'one', 'anyone', "questions", "someone", "child", "neighbour", "invesitigation", "screening", "assessment"], 
                                    bad_context=['not', 'mistakenly', 'likely', 'ie']):
        if good_match is None and bad_match is None:
            return -1
        if good_match is None:
            return 0
        # elif bad_match is None:
        #     return 1
        else:
            sub = good_match['subject']        
            ctx = good_match['context']
            if MentionPattern.lists_sim_enough(sub, bad_subjs, nlp) == 1:
                return 0
            if MentionPattern.lists_sim_enough(ctx, bad_context, nlp) == 1:
                return 0
            # return -1
            if bad_match is None:
                return 1
            else:
                return 1 if good_match['score'] * good_match['supports'] >= bad_match['score'] * bad_match['supports'] else 0

    @staticmethod
    def lists_sim_enough(l1, l2, nlp, threshold=0.8):
        if len(l1) == 0 or len(l2) == 0:
            return -1
        d1 = nlp(' '.join(l1))
        d2 = nlp(' '.join(l2))
        for t1 in d1:
            for t2 in d2:
                if t1.similarity(t2) > threshold:
                    return 1
        return 0

_nlp_lg = None

def get_nlp_lg():
    global _nlp_lg
    if _nlp_lg is None:
        _nlp_lg = spacy.load('en_core_web_lg')
    return _nlp_lg

