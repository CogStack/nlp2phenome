import utils
from annotation_docs import SemEHRAnnDoc, BasicAnn
import logging
from os.path import basename, isfile, join, split
from os import listdir, remove
import spacy

_spacy_nlp = None


def get_nlp_instance():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def get_sentences_as_anns(nlp, text):
    doc = nlp(text)
    anns = []
    for s in doc.sents:
        anns.append(BasicAnn(s.text, s.start_char, s.end_char))
    return anns


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
                if t.idx == pos:
                    token = t
        if token is not None:
            ta = TokenAbstraction(token, doc)
        else:
            return None
        return ta


class TokenAbstraction(object):
    def __init__(self, token, doc):
        self._t = token
        self._d = doc
        self._children = []
        self._root = None
        self._subject = None
        self._verbs = None
        self.do_abstract()

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

    def do_abstract(self):
        self._children = [t for t in self._t.children]
        t = self._t
        r = t
        while (t.head != t) and t.pos_ != u"VERB":
            t = t.head
            r = t
        if t is not None:
            self._verbs = [v for v in t.children if v.pos_ == u"VERB"]
            self._subject = [s for s in t.children if s.dep_ == u"nsubj"]
        self._root = r

    def to_dict(self):
        return {'children': [t.text for t in self.children], 'root': self.root.text, 'subject': [s.text for s in self.subject], 'verbs': [v.text for v in self.verbs]}


class ReportAbstractor(SemEHRAnnDoc):
    def __init__(self, ann_file):
        super(ReportAbstractor, self).__init__(ann_file)
        self._abstracted_sents = []

    def get_abstracted_sents(self):
        seq = 0
        for s in self.sentences:
            a_sent = AbstractedSentence(seq)
            seq += 1
            anns = sorted(self.annotations, key=lambda x: x.start)
            for a in anns:
                if a.overlap(s):
                    a_sent.add_token('%s%s[%s]' % ("%s: " % a.negation if a.negation == "Negated" else "", a.str, a.sty))
            self._abstracted_sents.append(a_sent)
            logging.debug(a_sent.tokens)


def test():
    ann_dir = 'C:/Users/hwu33/Downloads/working/semehr-runtime/radiology-reports/semehr_results/'
    files = [f for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
    for f in files:
        logging.debug('%s' % f)
        ra = ReportAbstractor(join(ann_dir, f))
        ra.get_abstracted_sents()
        logging.debug('\n')


def test_spacy():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(u"She said he might be getting better soon.")
    for token in doc:
        print(token.text, token.pos_, token.dep_, token.head.text, token.head.pos_,
              [child for child in token.children], token.idx, token.shape_)


def test_abstract_sentence():
    nlp = get_nlp_instance()
    abss = AbstractedSentence(1)
    abss.text = u"She said he might be getting better soon"
    result = abss.get_abstaction_by_pos(29, nlp)
    if result is not None:
        print(result.root, result.children, result.verbs, result.subject)


def test_sentences():
    nlp = get_nlp_instance()
    sents = get_sentences_as_anns(nlp, u"""
Circumstances leading to assessment.
Over the past week ZZZZZ.
    """)
    print([s.serialise_json() for s in sents])


if __name__ == "__main__":
    logging.basicConfig(level='DEBUG', format='[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s')
    # test_spacy()
    # test_abstract_sentence()
    test_sentences()
