import utils
from os import listdir
from os.path import basename, isfile, join
import logging
import re
from learners import LabelPerformance
# import reportreader as rr


class BasicAnn(object):
    """
    a simple NLP (Named Entity) annotation class
    """

    def __init__(self, str, start, end):
        self._str = str
        self._start = start
        self._end = end
        self._id = -1

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def str(self):
        return self._str

    @str.setter
    def str(self, value):
        self._str = value

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    def overlap(self, other_ann):
        if (other_ann.start <= self.start <= other_ann.end or other_ann.start <= self.end <= other_ann.end) or \
                (self.start <= other_ann.start <= self.end or self.start <= other_ann.end <= self.end):
            return True
        else:
            return False

    def is_larger(self, other_ann):
        return self.start <= other_ann.start and self.end >= other_ann.end \
               and not (self.start == other_ann.start and self.end == other_ann.end)

    def serialise_json(self):
        return {'start': self.start, 'end': self.end, 'str': self.str, 'id': self.id}

    @staticmethod
    def deserialise(jo):
        ann = BasicAnn(jo['start'], jo['start'], jo['end'])
        ann.id = jo['id']
        return ann


class EDIRAnn(BasicAnn):
    """
    EDIR annotation class
    """

    def __init__(self, str, start, end, type):
        self._type = type
        super(EDIRAnn, self).__init__(str, start, end)
        self._negated = False

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def negated(self):
        return self._negated

    @negated.setter
    def negated(self, value):
        self._negated = value

    @property
    def label(self):
        t = self.type
        if self.negated:
            t = 'neg_' + t
        return t


class ContextedAnn(BasicAnn):
    """
    a contextulised annotation class (negation/tempolarity/experiencer)
    """

    def __init__(self, str, start, end, negation, temporality, experiencer):
        self._neg = negation
        self._temp = temporality
        self._exp = experiencer
        super(ContextedAnn, self).__init__(str, start, end)

    @property
    def negation(self):
        return self._neg

    @negation.setter
    def negation(self, value):
        self._neg = value

    @property
    def temporality(self):
        return self._temp

    @temporality.setter
    def temporality(self, value):
        self._temp = value

    @property
    def experiencer(self):
        return self._exp

    @experiencer.setter
    def experiencer(self, value):
        self._exp = value


class PhenotypeAnn(ContextedAnn):
    """
    a simple customisable phenotype annotation (two attributes for customised attributes)
    """

    def __init__(self, str, start, end,
                 negation, temporality, experiencer,
                 major_type, minor_type):
        super(PhenotypeAnn, self).__init__(str, start, end, negation, temporality, experiencer)
        self._major_type = major_type
        self._minor_type = minor_type

    @property
    def major_type(self):
        return self._major_type

    @major_type.setter
    def major_type(self, value):
        self._major_type = value

    @property
    def minor_type(self):
        return self._minor_type

    @minor_type.setter
    def minor_type(self, value):
        self._minor_type = value

    def to_dict(self):
        return {
            'str': self.str,
            'start': self.start,
            'end': self.end,
            'negation': self.negation,
            'temporality': self.temporality,
            'experiencer': self.experiencer,
            'majorType': self.major_type,
            'minorType': self.minor_type
        }

    def serialise_json(self):
        dict = super(PhenotypeAnn, self).serialise_json()
        dict['major_type'] = self.major_type
        dict['minor_type'] = self.minor_type
        return dict

    @staticmethod
    def deserialise(jo):
        ann = PhenotypeAnn(jo['str'], jo['start'], jo['end'], jo['negation'], jo['temporality'],
                           jo['experiencer'], jo['major_type'], jo['minor_type'])
        ann.id = jo['id']
        return ann


class SemEHRAnn(ContextedAnn):
    """
    SemEHR Annotation Class
    """

    def __init__(self, str, start, end,
                 negation, temporality, experiencer,
                 cui, sty, pref, ann_type):
        super(SemEHRAnn, self).__init__(str, start, end, negation, temporality, experiencer)
        self._cui = cui
        self._sty = sty
        self._pref = pref
        self._ann_type = ann_type
        self._ruled_by = []

    @property
    def ruled_by(self):
        return self._ruled_by

    @property
    def cui(self):
        return self._cui

    @cui.setter
    def cui(self, value):
        self._cui = value

    @property
    def sty(self):
        return self._sty

    @sty.setter
    def sty(self, value):
        self._sty = value

    @property
    def ann_type(self):
        return self._ann_type

    @ann_type.setter
    def ann_type(self, value):
        self._ann_type = value

    @property
    def pref(self):
        return self._pref

    @pref.setter
    def pref(self, value):
        self._pref = value

    @staticmethod
    def deserialise(jo):
        ann = SemEHRAnn(jo['str'], jo['start'], jo['end'], jo['negation'], jo['temporality'],
                        jo['experiencer'], jo['cui'], jo['sty'], jo['pref'], 'mention')
        ann.id = jo['id']
        if 'ruled_by' in jo:
            ann._ruled_by = jo['ruled_by']
        if 'study_concepts' in jo:
            ann._study_concepts = jo['study_concepts']
        return ann


class SemEHRAnnDoc(object):
    """
    SemEHR annotation Doc
    """

    def __init__(self, file_path, ann_doc=None):
        if ann_doc is not None:
            self._doc = ann_doc
        else:
            self._doc = utils.load_json_data(file_path)
        self._anns = []
        self._phenotype_anns = []
        self._sentences = []
        self._others = []
        self.load_anns()

    def load_anns(self):
        all_anns = self._anns
        panns = self._phenotype_anns
        if 'sentences' in self._doc:
            # is a SemEHRAnnDoc serialisation
            self._anns = [SemEHRAnn.deserialise(a) for a in self._doc['annotations']]
            if 'phenotypes' in self._doc:
                self._phenotype_anns = [PhenotypeAnn.deserialise(a) for a in self._doc['phenotypes']]
            self._sentences = [BasicAnn.deserialise(a) for a in self._doc['sentences']]
        else:
            for anns in self._doc['annotations']:
                for ann in anns:
                    t = ann['type']
                    if t == 'Mention':
                        a = SemEHRAnn(ann['features']['string_orig'],
                                      int(ann['startNode']['offset']),
                                      int(ann['endNode']['offset']),

                                      ann['features']['Negation'],
                                      ann['features']['Temporality'],
                                      ann['features']['Experiencer'],

                                      ann['features']['inst'],
                                      ann['features']['STY'],
                                      ann['features']['PREF'],
                                      t)
                        all_anns.append(a)
                        a.id = 'cui-%s' % len(all_anns)
                    elif t == 'Phenotype':
                        a = PhenotypeAnn(ann['features']['string_orig'],
                                         int(ann['startNode']['offset']),
                                         int(ann['endNode']['offset']),

                                         ann['features']['Negation'],
                                         ann['features']['Temporality'],
                                         ann['features']['Experiencer'],

                                         ann['features']['majorType'],
                                         ann['features']['minorType'])
                        panns.append(a)
                        a.id = 'phe-%s' % len(panns)
                    elif t == 'Sentence':
                        a = BasicAnn('Sentence',
                                     int(ann['startNode']['offset']),
                                     int(ann['endNode']['offset']))
                        self._sentences.append(a)
                        self._sentences = sorted(self._sentences, key=lambda x: x.start)
                        a.id = 'sent-%s' % len(self._sentences)
                    else:
                        self._others.append(ann)

            sorted(all_anns, key=lambda x: x.start)

    @property
    def annotations(self):
        return self._anns

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, value):
        self._sentences = value

    @property
    def phenotypes(self):
        return self._phenotype_anns

    def learn_mappings_from_labelled(self, labelled_doc, lbl2insts, lbl2missed):
        ed = labelled_doc
        sd = self
        for e in ed.get_ess_entities():
            matched = False
            for a in sd.annotations:
                if a.overlap(e) and not e.is_larger(a):
                    matched = True
                    if e.type not in lbl2insts:
                        lbl2insts[e.type] = set()
                    lbl2insts[e.type].add('\t'.join([a.cui, a.pref, a.sty]))
                    continue
            # if not matched:
            if True:
                if e.type not in lbl2missed:
                    lbl2missed[e.type] = []
                lbl2missed[e.type].append(e.str.lower())

    @staticmethod
    def keep_max_len_anns(anns):
        ann2remove = set()
        for idx in range(len(anns)):
            a = anns[idx]
            for ni in range(idx + 1, len(anns)):
                b = anns[ni]
                if a.overlap(b):
                    if a.is_larger(b):
                        ann2remove.add(b)
                    elif b.is_larger(a):
                        ann2remove.add(a)
        for a in ann2remove:
            anns.remove(a)


class Concept2Mapping(object):
    """
    a mapping from annotations to phenotypes
    """

    def __init__(self, concept_map_file):
        self._concept_map_file = concept_map_file
        self._cui2label = {}
        self._concept2label = None
        self._type2concept = {}
        self._type2gaz = {}
        self._all_entities = []
        self.load_concept_mappings()

    def load_concept_mappings(self):
        concept_mapping = utils.load_json_data(self._concept_map_file)
        concept2types = {}
        for t in concept_mapping:
            self._type2concept[t] = []
            for text in concept_mapping[t]:
                c = text[:8]  # only to get the CUI
                self._type2concept[t].append(c)
                arr = text.split('\t')
                self._cui2label[c] = arr[1]
                if c not in concept2types:
                    concept2types[c] = []
                concept2types[c].append(t)
                self._all_entities.append(c.lower())
        self._concept2label = concept2types

    def load_gaz_dir(self, gaz_dir):
        files = [f for f in listdir(gaz_dir) if isfile(join(gaz_dir, f))]
        for f in files:
            if f.endswith('.lst'):
                t = f.split('.')[0]
                self._type2gaz[t] = utils.read_text_file(join(gaz_dir, f))
                self._all_entities += [t.lower() for t in self._type2gaz[t]]

    @property
    def cui2label(self):
        return self._cui2label

    @property
    def concept2label(self):
        return self._concept2label

    @concept2label.setter
    def concept2label(self, value):
        self._concept2label = value

    def type2cocnepts(self, type):
        return self._type2concept[type]

    @property
    def type2gaz(self):
        return self._type2gaz

    @property
    def all_entities(self):
        return self._all_entities


class CustomisedRecoginiser(SemEHRAnnDoc):
    """
    recognise target labels based on identified UMLS entities and
    customised labels
    """

    def __init__(self, file_path, concept_mapping, ann_doc=None):
        super(CustomisedRecoginiser, self).__init__(file_path=file_path, ann_doc=ann_doc)
        self._concept_mapping = concept_mapping
        self._mapped = None
        self._phenotypes = None
        self._combined = None
        self._full_text_folder = None
        self._full_text_file_pattern = '%s.txt'
        self._full_text = None

    @property
    def full_text_folder(self):
        return self._full_text_folder

    @full_text_folder.setter
    def full_text_folder(self, value):
        self._full_text_folder = value

    @property
    def full_text_file_pattern(self):
        return self._full_text_file_pattern

    @full_text_file_pattern.setter
    def full_text_file_pattern(self, value):
        self._full_text_file_pattern = value

    @property
    def concept2label(self):
        return self._concept_mapping.concept2label

    def get_mapped_labels(self):
        if self._mapped is not None:
            return self._mapped
        mapped = []
        for ann in self.annotations:
            if ann.cui in self.concept2label:
                for t in self.concept2label[ann.cui]:
                    ea = EDIRAnn(ann.str, ann.start, ann.end, t)
                    ea.negated = ann.negation == 'Negated'
                    ea.id = ann.id
                    mapped.append(ea)
        self._mapped = mapped
        return mapped

    def get_customised_phenotypes(self):
        if self._phenotypes is not None:
            return self._phenotypes
        self._phenotypes = []
        for ann in self.phenotypes:
            ea = EDIRAnn(ann.str, ann.start, ann.end, ann.minor_type)
            ea.negated = ann.negation == 'Negated'
            ea.id = ann.id
            self._phenotypes.append(ea)
        return self._phenotypes

    def get_ann_sentence(self, ann):
        sent = None
        for s in self.sentences:
            if ann.overlap(s):
                sent = s
                break
        if sent is None:
            print('sentence not found for %s' % ann.__dict__)
            return None
        return sent

    def get_previous_sentences(self, ann, include_self=True):
        sent = self.get_ann_sentence(ann)
        if sent is None:
            return None
        sents = []
        for s in self.sentences:
            if s.start < sent.start:
                sents.append(s)
        return sorted(sents + ([] if not include_self else [sent]), key=lambda s: s.start)

    def get_sent_anns(self, sent, ann_ignore=None, filter_fun=None, filter_param=None):
        ret = {'umls': [], 'phenotype': []}
        for a in self.annotations:
            if a.overlap(sent):
                if ann_ignore is not None and ann_ignore.overlap(a):
                    continue
                if filter_fun is not None and filter_fun(a, filter_param):
                    continue
                ret['umls'].append(a)
        for a in self.phenotypes:
            if a.overlap(sent):
                if ann_ignore is not None and ann_ignore.overlap(a):
                    continue
                if filter_fun is not None and filter_fun(a, filter_param):
                    continue
                ret['phenotype'].append(a)
        return ret

    def get_same_sentence_anns(self, ann):
        sent = self.get_ann_sentence(ann)
        if sent is None:
            return None
        return self.get_sent_anns(sent, ann)

    def get_prior_anns(self, ann, filter_fun=None, filter_param=None, contenxt_depth=-1):
        sents = self.get_previous_sentences(ann)
        ret = {'umls': [], 'phenotype': []}
        for s in sents[contenxt_depth:]:
            r = self.get_sent_anns(s, ann_ignore=ann, filter_fun=filter_fun, filter_param=filter_param)
            ret['umls'] += r['umls']
            ret['phenotype'] += r['phenotype']
        return ret

    def get_containing_anns(self, ann):
        c_anns = []
        for a in self.phenotypes:
            if ann != a and ann.str.lower() in a.str.lower() and len(a.str) > len(ann.str):
                c_anns.append(a)
        return c_anns

    @property
    def full_text(self):
        return self._full_text

    @full_text.setter
    def full_text(self, value):
        self._full_text = value

    def get_full_text(self, fk):
        if self._full_text is None and self._full_text_folder is not None and self._full_text_file_pattern is not None:
            self._full_text = utils.read_text_file_as_string(
                join(self._full_text_folder,
                     self._full_text_file_pattern % fk), encoding='utf-8')
        return self._full_text

    def relocate_all_anns(self, fk):
        t = self.get_full_text(fk)
        for a in self.phenotypes + self.annotations:
            s, e = relocate_annotation_pos(t, a.start, a.end, a.str)
            a.start = s
            a.end = e

    def re_segment_sentences(self, fk):
        text = self.get_full_text(fk)
        if text is not None:
            self.sentences = rr.get_sentences_as_anns(rr.get_nlp_instance(), text)

    def get_context_words(self, ann, file_key, n_words=2):
        sent = self.get_ann_sentence(ann)
        t = self.get_full_text(file_key)
        words = []
        if t is not None:
            s = t[sent.start:sent.end]
            context_start = ann.start - sent.start + len(ann.str)
            str = s[context_start:]
            p = re.compile(r'\[A-Za-z]{0,2}\b(\w+)\b')
            idx = 0
            for m in p.finditer(str):
                if idx <= n_words - 1:
                    words.append(str[m.span(1)[0]:m.span(1)[1]])
                else:
                    break
                idx += 1

            # use dependency tree to get context words
            # abss = rr.AbstractedSentence(1)
            # abss.text = s
            # result = abss.get_abstaction_by_pos(abss.locate_pos(ann.str), rr.get_nlp_instance())
            # dep_words = []
            # if result is not None:
            #     # subject
            #     dep_words.append(result.subject[0].text if len(result.subject) > 0 else 'empty')

            #     # first verb other than root verb
            #     dep_words.append(result.verbs[0].text if len(result.verbs) > 0 else 'empty')

            #     # root verb
            #     dep_words.append(result.root.text if result.root is not None else 'empty')

            #     # first child
            #     dep_words.append(result.children[0].text if len(result.children) > 0 else 'empty')
            # else:
            #     dep_words += ['empty'] *4
            #     logging.debug('not found [%s]' % s)
            # words += dep_words
        if len(words) == 0:
            words = ['empty']
        return words

    def get_anns_by_label(self, label, ignore_mappings=[], no_context=False):
        anns = []
        t = label.replace('neg_', '')
        for a in self.annotations:
            if a.cui not in self.concept2label:
                continue
            if a.cui in ignore_mappings:
                continue
            if len(a.ruled_by) > 0:
                continue
            if t in self.concept2label[a.cui]:
                if no_context:
                    anns.append(a)
                elif label.startswith('neg_') and a.negation == 'Negated':
                    anns.append(a)
                elif not label.startswith('neg_') and a.negation != 'Negated':
                    anns.append(a)
        # anns = []
        phenotypes = []
        smaller_to_remove = []
        for a in self.phenotypes:
            if a.minor_type == t:
                if a.str.lower() in [s.lower() for s in ignore_mappings]:
                    continue
                if no_context or (label.startswith('neg_') and a.negation == 'Negated') or \
                        (not label.startswith('neg_') and a.negation != 'Negated'):
                    overlaped = False
                    for ann in anns + phenotypes:
                        if ann.overlap(a):
                            if a.is_larger(ann):
                                smaller_to_remove.append(ann)
                            else:
                                overlaped = True
                                break
                    if not overlaped:
                        phenotypes.append(a)
        for o in smaller_to_remove:
            if o in anns:
                anns.remove(o)
            if o in phenotypes:
                phenotypes.remove(o)
        return anns + phenotypes

    def get_combined_anns(self):
        if self._combined is not None:
            return self._combined
        anns = [] + self.get_mapped_labels()
        for ann in self.get_customised_phenotypes():
            overlaped = False
            for m in self.get_mapped_labels():
                if ann.overlap(m):
                    overlaped = True
                    break
            if not overlaped:
                anns.append(ann)
        self._combined = anns
        return anns

    def validate_mapped_performance(self, gold_anns, label2performance):
        CustomisedRecoginiser.validate(gold_anns, self.get_mapped_labels(), label2performance)

    def validate_combined_performance(self, gold_anns, label2performance):
        CustomisedRecoginiser.validate(gold_anns,
                                       self.get_combined_anns(),
                                       label2performance)

    @staticmethod
    def validate(gold_anns, learnt_anns, label2performance):
        matched_ann_ids = []
        for ga in gold_anns:
            l = ga.label
            if l not in label2performance:
                label2performance[l] = LabelPerformance(l)
            performance = label2performance[l]
            matched = False
            for la in learnt_anns:
                if la.label == l and la.overlap(ga):
                    matched = True
                    performance.increase_true_positive()
                    matched_ann_ids.append(la.id)
                    break
            if not matched:
                performance.increase_false_negative()
        for la in learnt_anns:
            if la.id not in matched_ann_ids:
                l = la.label
                if l not in label2performance:
                    label2performance[l] = LabelPerformance(l)
                performance = label2performance[l]
                performance.increase_false_positive()

    @staticmethod
    def print_performances(label2performances):
        s = ''.join(['*' * 10, 'performance', '*' * 10])
        s += '\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ('label', 'precision', 'recall', 'f1', '#insts', 'false positive',
                                                     'false negative', 'true positive')
        for t in label2performances:
            p = label2performances[t]
            s += '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (t, p.precision, p.recall, p.f1,
                                                       p.true_positive + p.false_negative,
                                                       p.false_positive, p.false_negative, p.true_positive)
        logging.getLogger('performance').info(s)
        return s


def relocate_annotation_pos(t, s, e, string_orig):
    if t[s:e] == string_orig:
        return [s, e]
    candidates = []
    ito = re.finditer(r'[\s\.;\,\?\!\:\/$^](' + string_orig + r')[\s\.;\,\?\!\:\/$^]',
                      t, re.IGNORECASE)
    for mo in ito:
        # print mo.start(1), mo.end(1), mo.group(1)
        candidates.append({'dis': abs(s - mo.start(1)), 's': mo.start(1), 'e': mo.end(1), 'matched': mo.group(1)})
    if len(candidates) == 0:
        return [s, e]
    candidates.sort(cmp=lambda x1, x2: x1['dis'] - x2['dis'])
    # print candidates[0]
    return [candidates[0]['s'], candidates[0]['e']]