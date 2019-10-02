import utils
from nlp_to_phenome import SemEHRAnnDoc
import logging
from os.path import basename, isfile, join, split
from os import listdir, remove


class AbstractedSentence(object):
    def __init__(self, seq):
        self._seq = 0
        self._abstracted_tokens = []

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


if __name__ == "__main__":
    logging.basicConfig(level='DEBUG', format='[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s')
    ann_dir = 'C:/Users/hwu33/Downloads/working/semehr-runtime/radiology-reports/semehr_results/'
    files = [f for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
    for f in files:
        logging.debug('%s' % f)
        ra = ReportAbstractor(join(ann_dir, f))
        ra.get_abstracted_sents()
        logging.debug('\n')
