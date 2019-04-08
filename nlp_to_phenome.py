#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nlp2phenome
using AI models to infer patient phenotypes from identified named entities (instances of biomedical concepts)
"""

import xml.etree.ElementTree as ET
import utils
from os.path import basename, isfile, join, split
from os import listdir, remove
import json
import joblib as jl
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier, KDTree
from sklearn.metrics.pairwise import cosine_similarity
import graphviz
import numpy
import logging
import re


class EDIRDoc(object):
    """
    a class for reading EDIR annotation doc (XML)
    """
    def __init__(self, file_path):
        self._path = file_path
        self._root = None
        self._full_text = None
        self._word_offset_start = -1
        self._entities = None
        self.load()

    @property
    def file_path(self):
        return self._path

    def load(self):
        if not isfile(self.file_path):
            logging.debug('%s is NOT a file' % self.file_path)
            return
        tree = ET.parse(self.file_path)
        self._root = tree.getroot()
        self.get_word_offset_start()

    @property
    def get_full_text(self):
        if self._full_text is not None:
            return self._full_text
        if self._root is None:
            self.load()
        root = self._root
        d = ''
        start_offset = -1
        for p in root.findall('.//p'):
            for s in p:
                if 'proc' in s.attrib: # and s.attrib['proc'] == 'yes':
                    for w in s:
                        id_val = int(w.attrib['id'][1:])
                        if start_offset == -1:
                            start_offset = id_val
                        offset = id_val - start_offset
                        d += ' ' * (offset - len(d)) + w.text
        self._full_text = d
        return d

    def get_word_offset_start(self):
        if self._word_offset_start >= 0:
            return self._word_offset_start
        root = self._root
        offset_start = -1
        for e in root.findall('.//p/s[@proc]/w'):
            if 'id' not in e.attrib:
                continue
            else:
                offset_start = int(e.attrib['id'][1:])
                break
        if offset_start == -1:
            logging.debug('%s offset start could not be found' % self.file_path)
        self._word_offset_start = offset_start

    def get_ess_entities(self):
        if self._entities is not None:
            return self._entities
        root = self._root
        offset_start = self.get_word_offset_start()
        entities = []
        for e in root.findall('.//standoff/ents/ent'):
            if 'type' not in e.attrib:
                continue
            ent_type = e.attrib['type']
            if ent_type.startswith('label:'):
                continue
            negated = False
            if 'neg_' in ent_type:
                negated = True
                ent_type = ent_type.replace(r'neg_', '')
            str = ' '.join([part.text for part in e.findall('./parts/part')])
            ent_start = -1
            ent_end = -1
            for part in e.findall('./parts/part'):
                ent_start = int(part.attrib['sw'][1:]) - offset_start
                ent_end = ent_start + len(part.text)
            ann = EDIRAnn(str=str, start=ent_start, end=ent_end, type=ent_type)
            ann.negated = negated
            ann.id = len(entities)
            entities.append(ann)
        self._entities = entities
        return self._entities


class ConllDoc(EDIRDoc):
    """
    for Conll output from classification results
    """
    def __init__(self, file_path):
        super(ConllDoc, self).__init__(file_path)
        self._tokens = None
        self._label_white_list = None

    def set_label_white_list(self, labels):
        self._label_white_list = labels

    @property
    def conll_output(self):
        try:
            return '\n'.join([' '.join([t['t'], str(len(t['predicted_label'])), t['gold_label'],
                                        (('B-' if t['predicted_label'][-1]['ann'].start == t['offset'] else 'I-') +
                                         t['predicted_label'][-1]['label'] )
                                        if len(t['predicted_label']) > 0 else 'O'])
                              for t in self.get_token_list()])
        except:
            logging.error('processing [%s] failed' % self.file_path)
            return ''

    def get_token_list(self):
        if self._tokens is not None:
            return self._tokens
        self._tokens = []
        start_offset = -1
        root = self._root
        work_ess = list(self.get_ess_entities())
        matched_ess = set()
        for p in root.findall('.//p'):
            for s in p:
                if 'proc' in s.attrib: # and s.attrib['proc'] == 'yes':
                    for w in s:
                        id_val = int(w.attrib['id'][1:])
                        if start_offset == -1:
                            start_offset = id_val
                        offset = id_val - start_offset
                        token = {'t': w.text, 'id': w.attrib['id'], 'offset': offset,
                                 'gold_label': 'O', 'predicted_label': []}
                        for e in work_ess:
                            label = e.type.replace('neg_', '').lower().strip()
                            if self._label_white_list is not None and label not in self._label_white_list:
                                continue
                            if token['offset'] == e.start:
                                token['gold_label'] = 'B-' + label
                                matched_ess.add(e)
                            elif e.start < token['offset'] < e.end:
                                token['gold_label'] = 'I-' + label
                                matched_ess.add(e)
                        self._tokens.append(token)
        left_ess = [e for e in work_ess if e not in matched_ess
                    and e.type.replace('neg_', '') in self._label_white_list]
        if len(left_ess) > 0:
            logging.error('leftovers: [%s] at %s' % (
                '\n'.join(['%s (%s,%s)' % (a.type, a.start, a.end) for a in left_ess]), self.file_path))
        return self._tokens

    def add_predicted_labels(self, predicted_label):
        """
        append prediction result to the doc, one annotation a time
        :param predicted_label: labelled ann {'label': ..., 'ann': ann object}
        :return:
        """
        if self._label_white_list is not None and predicted_label['label'] not in self._label_white_list:
            return
        for token in self.get_token_list():
            if predicted_label['ann'].start <= token['offset'] < predicted_label['ann'].end:
                token['predicted_label'].append(predicted_label)


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


class SemEHRAnnDoc(object):
    """
    SemEHR annotation Doc
    """
    def __init__(self, file_path):
        self._doc = utils.load_json_data(file_path)
        self._anns = []
        self._phenotype_anns = []
        self._sentences = []
        self._others = []
        self.load_anns()

    def load_anns(self):
        all_anns = self._anns
        panns = self._phenotype_anns
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
                    # SemEHRAnnDoc.keep_max_len_anns(panns)
                    a.id = 'phe-%s' % len(panns)
                elif t == 'Sentence':
                    a = BasicAnn('Sentence',
                                 int(ann['startNode']['offset']),
                                 int(ann['endNode']['offset']))
                    self._sentences.append(a)
                    a.id = 'sent-%s' % len(self._sentences)
                else:
                    self._others.append(ann)
        SemEHRAnnDoc.keep_max_len_anns(panns)
        sorted(all_anns, key=lambda x: x.start)

    @property
    def annotations(self):
        return self._anns

    @property
    def sentences(self):
        return self._sentences

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
        for idx in xrange(len(anns)):
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
            self._type2concept[t] =[]
            for text in concept_mapping[t]:
                c = text[:8] # only to get the CUI
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
    def __init__(self, file_path, concept_mapping):
        super(CustomisedRecoginiser, self).__init__(file_path=file_path)
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
            print 'sentence not found for %s' % ann.__dict__
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
            if ann!=a and ann.str.lower() in a.str.lower() and len(a.str) > len(ann.str):
                c_anns.append(a)
        return c_anns

    def get_full_text(self, fk):
        if self._full_text is None and self._full_text_folder is not None and self._full_text_file_pattern is not None:
            self._full_text = utils.read_text_file_as_string(
                join(self._full_text_folder,
                     self._full_text_file_pattern % fk))
        return self._full_text

    def get_context_words(self, ann, file_key, n_words=1):
        sent = self.get_ann_sentence(ann)
        t = self.get_full_text(file_key)
        words = []
        if t is not None:
            s = t[sent.start:sent.end]
            context_start = ann.start - sent.start + len(ann.str)
            str = s[context_start:]
            p = re.compile(r'\b(\w+)\b')
            idx = 0
            for m in p.finditer(str):
                if idx <= n_words - 1:
                    words.append(str[m.span(1)[0]:m.span(1)[1]])
                else:
                    break
                idx += 1
        if len(words) == 0:
            words =['empty']
        return words

    def get_anns_by_label(self, label, ignore_mappings=[], no_context=False):
        anns = []
        t = label.replace('neg_', '')
        for a in self.annotations:
            if a.cui not in self.concept2label:
                continue
            if a.cui in ignore_mappings:
                continue
            if t in self.concept2label[a.cui]:
                if no_context:
                    anns.append(a)
                elif label.startswith('neg_') and a.negation == 'Negated':
                    anns.append(a)
                elif not label.startswith('neg_') and a.negation != 'Negated':
                    anns.append(a)
        anns = []
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


class LabelModel(object):
    """
    a machine learning based class for inferring phenotypes from NLP results
    features:
    - feature weighing
    - transparent models
    """
    def __init__(self, label, max_dimensions=None):
        self._label = label
        self._lbl_data = {}
        self._cui2label = {}
        self._selected_dims = None
        self._max_dimensions = 2000 if max_dimensions == None else max_dimensions
        self._tps = 0
        self._fps = 0
        self._lbl_one_dimension = True
        self._lbl2tfidf_dims = {}
        self._label_dimensions = []
        self._rare_labels = {}
        self._lbl2classifiers = {}

    def get_binary_cluster_classifier(self, label):
        if label in self._lbl2classifiers:
            return self._lbl2classifiers[label]
        else:
            return None

    @property
    def cluster_classifier_dict(self):
        return self._lbl2classifiers

    def put_binary_cluster_classifier(self, label, classifier):
        self._lbl2classifiers[label] = classifier

    @property
    def rare_labels(self):
        return self._rare_labels

    def add_rare_label(self, label, tp_ratio):
        self._rare_labels[label] = tp_ratio

    @property
    def use_one_dimension_for_label(self):
        return self._lbl_one_dimension

    @use_one_dimension_for_label.setter
    def use_one_dimension_for_label(self, value):
        self._lbl_one_dimension = value

    @property
    def cui2label(self):
        return self._cui2label

    @property
    def label(self):
        return self._label

    def add_label_dimension(self, value):
        if value.lower() not in self._label_dimensions:
            self._label_dimensions.append(value.lower())
            # if tp is not None:
            #     self._tp_labels.add(value.lower())
            # if fp is not None:
            #     self._fp_labels.add(value.lower())

    def add_label_dimension_by_annotation(self, ann):
        self.add_label_dimension(LabelModel.get_ann_dim_label(ann, no_negation=True))

    def add_context_dimension(self, value, tp=None, fp=None, lbl='united'):
        if lbl not in self._lbl_data:
            self._lbl_data[lbl] = {'dims': [], 't2f': {}, 'tps': set(), 'fps': set()}
        d = self._lbl_data[lbl]
        if value.lower() not in d['dims']:
            d['dims'].append(value.lower())
        if value.lower() not in d['t2f']:
            d['t2f'][value.lower()] = 1
        else:
            d['t2f'][value.lower()] = d['t2f'][value.lower()] + 1
        tps = d['tps']
        fps = d['fps']
        if tp is not None:
            tps.add(value.lower())
        if fp is not None:
            fps.add(value.lower())

    def add_context_dimension_by_annotation(self, ann, tp=None, fp=None, lbl=None):
        self.add_context_dimension(LabelModel.get_ann_dim_label(ann, generalise=True, no_negation=True), tp=tp, fp=fp, lbl=lbl)

    def get_top_freq_dimensions(self, k, lbl='united'):
        if self._selected_dims is not None:
            return self._selected_dims
        if lbl not in self._lbl_data:
            return []
        l2f = self._lbl_data[lbl]['t2f']
        df = [(l, l2f[l]) for l in l2f]
        df = sorted(df, key=lambda x: -x[1])
        self._selected_dims = [d[0] for d in df[:k]]
        return self._selected_dims

    def get_top_tfidf_dimensions(self, k, lbl='united'):
        if lbl in self._lbl2tfidf_dims:
            return self._lbl2tfidf_dims[lbl]
        self._lbl2tfidf_dims[lbl] = {}
        if lbl not in self._lbl_data:
            logging.info('label [%s] has no contextual info' % lbl)
            return []
        d = self._lbl_data[lbl]
        tps = d['tps']
        fps = d['fps']
        idf_weight = 1.0
        if len(tps) > 0 and len(fps) > 0:
            idf_weight = 1.0 * len(tps) / len(fps)
        df = []
        max_score = 0
        for l in d['t2f']:
            idf = 1.0 / ( (1 if l in d['tps'] else 0) + (1 if l in d['fps'] else 0) )
            score = 1.0 * d['t2f'][l] / (len(tps) + len(fps))
            if idf_weight == 1 or (l in d['tps'] and l in d['fps']):
                score = score * idf
                # if l in d['tps'] and l in d['fps']:
                #     score *= 0.5
            elif l in d['fps']:
                score *= idf_weight * idf
            max_score = max(score, max_score)
            df.append((l, score))
        df = sorted(df, key=lambda x: -x[1])
        # logging.debug(df)
        self._lbl2tfidf_dims[lbl] = [(t[0], t[1] * 1.0 / max_score ) for t in df[:k]]
        logging.debug('%s ==> [%s]' % (lbl, self._lbl2tfidf_dims[lbl]))
        return self._lbl2tfidf_dims[lbl]

    @property
    def max_dimensions(self):
        return self._max_dimensions

    @max_dimensions.setter
    def max_dimensions(self, value):
        if value is None:
            self._max_dimensions = 2000
        self._max_dimensions = value

    @property
    def label_dimensions(self):
        return self._label_dimensions

    def context_dimensions(self, lbl):
        if lbl not in self._lbl_data:
            return []
        # logging.info('%s`s dims: %s' % (lbl, self._lbl_data[lbl]['dims']))
        return self._lbl_data[lbl]['dims']

    def encode_ann(self, ann, context_anns, lbl='united', extra_dims=None):
        ann_label = LabelModel.get_ann_dim_label(ann)
        encoded = []
        # if self.use_one_dimension_for_label:
        #     if ann_label in self.label_dimensions:
        #         encoded.append(self.label_dimensions.index(ann_label))
        #     else:
        #         encoded.append(-1)
        # else:
        #     for l in self.label_dimensions:
        #         if l == ann_label:
        #             encoded.append(1)
        #         else:
        #             encoded.append(0)
        context_labels = [LabelModel.get_ann_dim_label(ann, generalise=True, no_negation=True) for ann in context_anns]
        for l, score in self.get_top_tfidf_dimensions(self.max_dimensions, lbl=lbl): # self.context_dimensions:
            # freq = 0
            # for cl in context_labels:
            #     if cl.lower() == l.lower():
            #         freq += 1
            if l in context_labels:
                encoded.append(1)
            else:
                encoded.append(0)
            # encoded.append(freq * score)
        return encoded + ([] if extra_dims is None else extra_dims)

    def collect_dimensions(self, ann_dir):
        cm = Concept2Mapping(_concept_mapping)
        file_keys = [f.split('.')[0] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
        # collect dimension labels
        for fk in file_keys:
            cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), cm)
            t = self.label.replace('neg_', '')
            anns = cr.get_anns_by_label(t)
            neg_anns = cr.get_anns_by_label('neg_' + t)
            for a in anns + neg_anns:
                self.add_label_dimension_by_annotation(a)
                # self.add_context_dimension_by_annotation(a)
                if (a.negation != 'Negated' and self.label.startswith('neg_')) or \
                        (a.negation == 'Negated' and not self.label.startswith('neg_')):
                    continue
                sanns = cr.get_same_sentence_anns(a)
                context_anns = [] + sanns['umls'] + sanns['phenotype']
                #collect cui labels
                for u in sanns['umls']:
                    self._cui2label[u.cui] = u.pref
                for c in context_anns:
                    self.add_context_dimension_by_annotation(c)

    def collect_tfidf_dimensions(self, ann_dir, gold_dir, ignore_context=False, separate_by_label=False, full_text_dir=None):
        cm = Concept2Mapping(_concept_mapping)
        file_keys = [f.split('.')[0] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
        # collect dimension labels
        tp_freq = 0
        fp_freq = 0
        label_type = self.label.replace('neg_', '')
        fn_freq = 0
        for fk in file_keys:
            cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), cm)
            if full_text_dir is not None:
                cr.full_text_folder = full_text_dir
            gd = EDIRDoc(join(gold_dir, '%s-ann.xml' % fk))
            if not isfile(join(gold_dir, '%s-ann.xml' % fk)):
                continue
            t = self.label.replace('neg_', '')
            anns = cr.get_anns_by_label(t)
            neg_anns = cr.get_anns_by_label('neg_' + t)

            not_matched_gds = []
            for e in gd.get_ess_entities():
                if (ignore_context and e.label.replace('neg_', '') == label_type) \
                        or (not ignore_context and e.label == self.label):
                    not_matched_gds.append(e.id)
            for a in anns + neg_anns:
                # self.add_context_dimension_by_annotation(a)
                self.add_label_dimension_by_annotation(a)
                if (not ignore_context) and ((a.negation != 'Negated' and self.label.startswith('neg_')) or \
                        (a.negation == 'Negated' and not self.label.startswith('neg_'))):
                    logging.info('skipped because context')
                    continue

                matched = False
                for g in gd.get_ess_entities():
                    if g.id in not_matched_gds:
                        gt = g.label.replace('neg_', '')
                        if g.overlap(a) and ((g.label == self.label and not ignore_context) or
                                             (ignore_context and gt == label_type)):
                            matched = True
                            tp_freq += 1
                            not_matched_gds.remove(g.id)
                if not matched:
                    fp_freq += 1

                sanns = cr.get_prior_anns(a, contenxt_depth=-3)
                context_anns = [] + sanns['umls'] + sanns['phenotype'] + cr.get_context_words(a, fk)
                #collect cui labels
                for u in sanns['umls']:
                    self._cui2label[u.cui] = u.pref
                for c in context_anns:
                    self.add_context_dimension_by_annotation(c, tp=True if matched else None,
                                                             fp=True if not matched else None,
                                                             lbl='united' if not separate_by_label else
                                                             LabelModel.get_ann_query_label(a))
            fn_freq += len(not_matched_gds)
        self._tps = tp_freq
        self._fps = fp_freq
        logging.debug('tp: %s, fp: %s, fn: %s' % (tp_freq, fp_freq, fn_freq))

    def get_low_quality_labels(self, ann_dir, gold_dir, accurate_threshold=0.05, min_sample_size=20):
        return [t[0] for t in self.assess_label_quality(ann_dir, gold_dir)
                if t[1] <= accurate_threshold and t[2] + t[3] >= min_sample_size]

    def assess_label_quality(self, ann_dir, gold_dir, separate_by_label=True, ignore_context=True):
        if ignore_context:
            logging.info('doing learning without considering contextual info')
        # print self.get_top_tfidf_dimensions(self.max_dimensions)
        cm = Concept2Mapping(_concept_mapping)
        file_keys = [f.split('.')[0] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
        label_type = self.label.replace('neg_', '')
        query_label_perform = {}
        for fk in file_keys:
            cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), cm)
            if not isfile(join(gold_dir, '%s-ann.xml' % fk)):
                continue
            gd = EDIRDoc(join(gold_dir, '%s-ann.xml' % fk))

            not_matched_gds = []
            for e in gd.get_ess_entities():
                if (ignore_context and e.label.replace('neg_', '') == label_type) \
                        or (not ignore_context and e.label == self.label):
                    not_matched_gds.append(e.id)

            anns = cr.get_anns_by_label(self.label, no_context=ignore_context)
            for a in anns:
                multiple_true_positives = 0
                matched = False
                for g in gd.get_ess_entities():
                    if g.id in not_matched_gds:
                        gt = g.label.replace('neg_', '')
                        if g.overlap(a) and ((g.label == self.label and not ignore_context) or
                                             (ignore_context and gt == label_type)):
                            if matched:
                                multiple_true_positives += 1
                            matched = True
                            not_matched_gds.remove(g.id)

                if separate_by_label:
                    lbl = LabelModel.get_ann_query_label(a)
                else:
                    lbl = 'united'
                ql = lbl
                if ql not in query_label_perform:
                    query_label_perform[ql] = {'c': 0, 'w': 0}
                if matched:
                    query_label_perform[ql]['c'] += 1
                else:
                    query_label_perform[ql]['w'] += 1
        lbls = [(l,
                 1.0 * query_label_perform[l]['c'] / (query_label_perform[l]['c'] + query_label_perform[l]['w']),
                 query_label_perform[l]['c'],
                 query_label_perform[l]['w']) for l in query_label_perform]
        return sorted(lbls, key=lambda x : x[1])

    def load_data(self, ann_dir, gold_dir, verbose=True, ignore_mappings=[], ignore_context=False,
                  separate_by_label=False, ful_text_dir=None):
        if ignore_context:
            logging.info('doing learning without considering contextual info')
        # print self.get_top_tfidf_dimensions(self.max_dimensions)
        cm = Concept2Mapping(_concept_mapping)
        file_keys = [f.split('.')[0] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
        lbl2data = {}
        false_negatives = 0
        lbl2tps = {}
        label_type = self.label.replace('neg_', '')
        query_label_perform = {}
        for fk in file_keys:
            cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), cm)
            if ful_text_dir is not None:
                cr.full_text_folder = ful_text_dir
            if not isfile(join(gold_dir, '%s-ann.xml' % fk)):
                continue
            gd = EDIRDoc(join(gold_dir, '%s-ann.xml' % fk))

            not_matched_gds = []
            for e in gd.get_ess_entities():
                if (ignore_context and e.label.replace('neg_', '') == label_type) \
                        or (not ignore_context and e.label == self.label):
                    not_matched_gds.append(e.id)

            anns = cr.get_anns_by_label(self.label, ignore_mappings=ignore_mappings, no_context=ignore_context)
            for a in anns:
                multiple_true_positives = 0
                t2anns = cr.get_prior_anns(a)
                # if len(t2anns['umls']) + len(t2anns['phenotype']) == 0:
                #     t2anns = cr.get_prior_anns(a, contenxt_depth=-2)
                context_anns = [] + t2anns['umls'] + t2anns['phenotype'] + \
                               cr.get_context_words(a, fk)
                matched = False
                for g in gd.get_ess_entities():
                    if g.id in not_matched_gds:
                        gt = g.label.replace('neg_', '')
                        if g.overlap(a) and ((g.label == self.label and not ignore_context) or
                                             (ignore_context and gt == label_type)):
                            if matched:
                                multiple_true_positives += 1
                            matched = True
                            not_matched_gds.remove(g.id)
                if verbose:
                    if not matched:
                        logging.debug('%s %s %s' % ('!',
                                      self.get_ann_dim_label(a) +
                                      ' // ' + ' | '.join(self.get_ann_dim_label(a, generalise=True)
                                                          for a in context_anns), fk))
                    else:
                        logging.debug('%s %s %s' % ('R',
                                                    self.get_ann_dim_label(a) + ' // ' + ' | '.join(self.get_ann_dim_label(a, generalise=True)
                                                                  for a in context_anns), fk))
                if separate_by_label:
                    lbl = LabelModel.get_ann_query_label(a)
                else:
                    lbl = 'united'
                if lbl not in lbl2data:
                    lbl2data[lbl] = {'X': [], 'Y': [], 'multiple_tps': 0, 'doc_anns': []}
                X = lbl2data[lbl]['X']
                Y = lbl2data[lbl]['Y']
                lbl2data[lbl]['doc_anns'].append({'d': fk, 'ann': a, 'label': self.label})
                Y.append([1 if matched else 0])
                extra_dims = [1] if len(cr.get_containing_anns(a)) > 0 else [0]
                X.append(self.encode_ann(a, context_anns, lbl=lbl, extra_dims=extra_dims))
                lbl2data[lbl]['multiple_tps'] += multiple_true_positives
                ql = lbl
                if ql not in query_label_perform:
                    query_label_perform[ql] = {'c': 0, 'w': 0}
                if matched:
                    query_label_perform[ql]['c'] += 1
                else:
                    query_label_perform[ql]['w'] += 1
            false_negatives += len(not_matched_gds)

            missed = None
            for g in gd.get_ess_entities():
                if g.id in not_matched_gds:
                    missed = g
                    logging.debug('\t'.join(['M',  g.str, str(g.negated), str(g.start), str(g.end), join(gold_dir, '%s-ann.xml' % fk)]))
            # if len(not_matched_gds) > 0:
            #     print not_matched_gds
            #     for a in anns:
            #         logging.debug(a.str, a.start, a.end, missed.overlap(a))
        bad_labels = []
        for ql in query_label_perform:
            p = query_label_perform[ql]
            if p['c'] == 0 or (1.0 * p['w'] / p['c'] < 0.05):
                bad_labels.append(ql)
        return {'lbl2data': lbl2data,
                'fns': false_negatives, 'bad_labels': bad_labels, 'files': file_keys}

    def serialise(self, output_file):
        jl.dump(self, output_file)

    @staticmethod
    def type_related_ann_filter(ann, type):
        if hasattr(ann, 'cui'):
            return not ann.cui.lower() in _cm_obj.all_entities
            # return not ann.cui in _cm_obj.type2cocnepts(type)
        else:
            return not ann.str.lower() in  _cm_obj.all_entities
            # return not ann.str in _cm_obj.type2gaz[type]


    @staticmethod
    def get_ann_query_label(ann):
        # return ann.str.lower()
        neg = ''
        # if hasattr(ann, 'negation'):
        #     neg = 'neg_' if ann.negation == 'Negated' else ''
        # else:
        #     neg = 'neg_' if ann.negated else ''
        # if hasattr(ann, 'cui'):
        #     return neg + ann.cui + ' ' + str(ann.pref)
        # else:
        #     return neg + ann.str.lower()
        return neg + ann.str.lower()

    @staticmethod
    def deserialise(serialised_file):
        return jl.load(serialised_file)

    @staticmethod
    def get_ann_dim_label(ann, generalise=False, no_negation=False):
        if isinstance(ann, basestring):
            return 'WORD_%s' % ann
        negated = ''
        label = ann.str
        if (hasattr(ann, 'negation') and ann.negation == 'Negated') or (hasattr(ann, 'negated') and ann.negated):
            negated = 'neg_'
        if no_negation:
            negated = ''
        # if hasattr(ann, 'cui'):
        #     label = ann.cui + ' ' + str(ann.pref)
            # ann.str
        if hasattr(ann, 'minor_type'):
            label = ann.str
        # if generalise and hasattr(ann, 'sty'):
        #     label = ann.sty
            # if ann.sty.lower() == 'body part, organ, or organ component':
        negated = ''
        return negated + label.lower()
        # return ann.str.lower() if not isinstance(ann, SemEHRAnn) else ann.cui.lower()

    @staticmethod
    def decision_tree_learning(X, Y, lm, output_file=None, pca_dim=None, pca_file=None, tree_viz_file=None, lbl='united'):
        if len(X) <= _min_sample_size:
            logging.warning('not enough data found for prediction: %s' % lm.label)
            if isfile(output_file):
                remove(output_file)
            return
        pca = None
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim)
            X_new = pca.fit_transform(X)
        else:
            X_new = X
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_new, Y)
        if output_file is not None:
            jl.dump(clf, output_file)
            logging.info('model file saved to %s' % output_file)
        if pca is not None and pca_file is not None:
            jl.dump(pca, pca_file)
        if tree_viz_file is not None:
            label_feature_names = []
            if lm.use_one_dimension_for_label:
                label_feature_names.append('label')
            else:
                for l in lm.label_dimensions:
                    if l.upper() in lm.cui2label:
                        label_feature_names.append('lbl: ' + lm.cui2label[l.upper()])
                    else:
                        label_feature_names.append('lbl: ' + l.upper())
            dot_data = tree.export_graphviz(clf, out_file=None,
                                            filled=True, rounded=True,
                                            feature_names=label_feature_names +
                                                          [(str(lm.cui2label[l.upper()]) + '(' + l.upper() + ')') if l.upper() in lm.cui2label else l
                                                           for l in lm.context_dimensions(lbl)],
                                            class_names=['Yes', 'No'],
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(tree_viz_file)

    @staticmethod
    def random_forest_learning(X, Y, output_file=None):
        if len(X) == 0:
            logging.warning('no data found for prediction')
            return
        clf = RandomForestClassifier()
        clf = clf.fit(X, Y)
        if output_file is not None:
            jl.dump(clf, output_file)
            logging.info('model file saved to %s' % output_file)

    @staticmethod
    def svm_learning(X, Y, output_file=None):
        if len(X) == 0:
            logging.info('no data found for prediction')
            return
        v = -1
        all_same = True
        for y in Y:
            if v == -1:
                v = y[0]
            if v != y[0]:
                all_same = False
                break
        if all_same:
            logging.warning('all same labels %s' % Y)
            return
        clf = svm.SVC(kernel='sigmoid')
        clf = clf.fit(X, Y)
        if output_file is not None:
            jl.dump(clf, output_file)
            logging.info('model file saved to %s' % output_file)

    @staticmethod
    def gpc_learning(X, Y, output_file=None):
        gpc = GaussianProcessClassifier().fit(X, Y)
        if output_file is not None:
            jl.dump(gpc, output_file)
            logging.info('model file saved to %s' % output_file)

    @staticmethod
    def gaussian_nb(X, Y, output_file=None):
        gnb = GaussianNB().fit(X, Y)
        if output_file is not None:
            jl.dump(gnb, output_file)
            logging.info('model file saved to %s' % output_file)

    @staticmethod
    def cluster(X, Y, output_file=None):
        dbm = DBSCAN(eps=.50).fit(X)
        cls2label = {}
        for idx in xrange(len(dbm.labels_)):
            c = dbm.labels_[idx]
            cls = 'cls%s' % c
            if cls not in cls2label:
                cls2label[cls] = {'t': 0,  'f':0}
            if Y[idx] == [0]:
                cls2label[cls]['f'] += 1
            else:
                cls2label[cls]['t'] += 1
        logging.info(cls2label)
        kdt = KDTree(X)
        if output_file is not None:
            jl.dump({'dbm': dbm, 'X': X, 'Y': Y, 'kdt': kdt, 'cls2label': cls2label}, output_file)
            logging.info('complex model file saved to %s' % output_file)

    @staticmethod
    def cluster_predict(X, Y, fns, multiple_tps, model_file, performance, separate_performance=None):
        all_true = False
        if not isfile(model_file):
            logging.info('model file NOT FOUND: %s' % model_file)
            all_true = True
        else:
            m = jl.load(model_file)
            dbm = m['dbm']
            kdt = m['kdt']
            P = m.predict(X)
            if fns > 0:
                logging.debug('missed instances: %s' % fns)
                performance.increase_false_negative(fns)
            if multiple_tps > 0:
                performance.increase_true_positive(multiple_tps)
        if all_true or len(X) <= _min_sample_size:
            logging.warn('using querying instead of predicting')
            P = numpy.ones(len(X))
        else:
            logging.info('instance size %s' % len(P))
        for idx in xrange(len(P)):
            if P[idx] == Y[idx]:
                if P[idx] == 1.0:
                    performance.increase_true_positive()
                    if separate_performance is not None:
                        separate_performance.increase_true_positive()
            elif P[idx] == 1.0:
                performance.increase_false_positive()
                if separate_performance is not None:
                    separate_performance.increase_false_positive()
            else:
                performance.increase_false_negative()
                if separate_performance is not None:
                    separate_performance.increase_false_negative()

    @staticmethod
    def knn_classify(X, Y, output_file=None):
        knn = KNeighborsClassifier(n_neighbors=2).fit(X, Y)
        if output_file is not None:
            jl.dump(knn, output_file)
            logging.info('model file saved to %s' % output_file)

    @staticmethod
    def predict_use_simple_stats(tp_ratio, Y, multiple_tps, performance, ratio_cut_off=0.15, separate_performance=None,
                                 id2conll=None, doc_anns=None, file_pattern=None, doc_folder=None,
                                 label_whitelist=None):
        P = numpy.ones(len(Y)) if tp_ratio >= ratio_cut_off else numpy.zeros(len(Y))
        if multiple_tps > 0:
            performance.increase_true_positive(multiple_tps)
            if separate_performance is not None:
                separate_performance.increase_true_positive(multiple_tps)
        LabelModel.cal_performance(P, Y, performance, separate_performance,
                                   id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                                   doc_folder=doc_folder,
                                   label_whitelist=label_whitelist)

    @staticmethod
    def cal_performance(P, Y, performance, separate_performance=None,
                        id2conll=None, doc_anns=None, file_pattern=None, doc_folder=None, label_whitelist=None):
        doc2predicted = {}
        for idx in xrange(len(P)):
            if P[idx] == Y[idx]:
                if P[idx] == 1.0:
                    performance.increase_true_positive()
                    if separate_performance is not None:
                        separate_performance.increase_true_positive()
            elif P[idx] == 1.0:
                performance.increase_false_positive()
                if separate_performance is not None:
                    separate_performance.increase_false_positive()
            else:
                performance.increase_false_negative()
                if separate_performance is not None:
                    separate_performance.increase_false_negative()
            if P[idx] == 1.0 and id2conll is not None and doc_anns is not None and doc_folder is not None:
                d = doc_anns[idx]['d']
                labeled_ann = {'label': doc_anns[idx]['label'],
                               'ann': doc_anns[idx]['ann']}
                if d not in doc2predicted:
                    doc2predicted[d] = [labeled_ann]
                else:
                    doc2predicted[d].append(labeled_ann)
        for d in doc2predicted:
            if d not in id2conll:
                id2conll[d] = ConllDoc(join(doc_folder, file_pattern % d))
                if label_whitelist is not None:
                    id2conll[d].set_label_white_list(label_whitelist)
            cnll = id2conll[d]
            for anns in doc2predicted[d]:
                cnll.add_predicted_labels(anns)


    @staticmethod
    def predict_use_model(X, Y, fns, multiple_tps, model_file, performance,
                          pca_model_file=None, separate_performance=None,
                          id2conll=None, doc_anns=None, file_pattern=None, doc_folder=None, label_whitelist=None):
        all_true = False
        if not isfile(model_file):
            logging.info('model file NOT FOUND: %s' % model_file)
            all_true = True
        else:
            if pca_model_file is not None:
                pca = jl.load(pca_model_file)
                X_new = pca.transform(X)
            else:
                X_new = X
            m = jl.load(model_file)
            P = m.predict(X_new)
            if fns > 0:
                logging.debug('missed instances: %s' % fns)
                performance.increase_false_negative(fns)
            if multiple_tps > 0:
                performance.increase_true_positive(multiple_tps)
                if separate_performance is not None:
                    separate_performance.increase_true_positive(multiple_tps)
        if all_true: # or len(X) <= _min_sample_size:
            logging.warn('using querying instead of predicting')
            P = numpy.ones(len(X))
        else:
            logging.info('instance size %s' % len(P))
        LabelModel.cal_performance(P, Y, performance, separate_performance,
                                   id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                                   doc_folder=doc_folder, label_whitelist=label_whitelist)


class BinaryClusterClassifier(object):
    def __init__(self, label):
        self._name = label
        self._class1reps = None
        self._class2reps = None

    @property
    def class1reps(self):
        return self._class1reps

    @property
    def class2reps(self):
        return self._class2reps

    def cluster(self, class1_data, class2_data):
        self._class1reps = BinaryClusterClassifier.do_clustering(class1_data, class_prefix='cls1:')
        self._class2reps = BinaryClusterClassifier.do_clustering(class2_data, class_prefix='cls2:')

    def classify(self, x, threshold=0.5, complementary_classifiers=None):
        p = BinaryClusterClassifier.calculate_most_similar(self, x)
        mp = p
        if p[1] < threshold and complementary_classifiers is not None:
            for classifer in complementary_classifiers:
                logging.debug('do extra classifying when the similarity is too low ...')
                p = BinaryClusterClassifier.calculate_most_similar(classifer, x)
                logging.debug('extra result @ %s' % p[1])
                mp = p if p[1] > mp[1] else mp
                if p[1] > threshold:
                    # stop when once exceeding the threshold
                    break
        return mp, 0 if mp[0].startswith('cls2:') else 1

    @staticmethod
    def calculate_most_similar(classifier, x):
        results = []
        xa = numpy.array(x).reshape(1, -1)
        for cls in classifier.class1reps:
            results.append((cls, cosine_similarity(xa, classifier.class1reps[cls])))
        for cls in classifier.class2reps:
            results.append((cls, cosine_similarity(xa, classifier.class2reps[cls])))
        return sorted(results, key=lambda x: -x[1])[0]

    @staticmethod
    def do_clustering(X, class_prefix='cls:'):
        dbm = DBSCAN(eps=1.0).fit(X)
        cls2insts = {}
        for idx in xrange(len(dbm.labels_)):
            c = dbm.labels_[idx]
            cls = '%s%s' % (class_prefix, c)
            if cls not in cls2insts:
                cls2insts[cls] = [X[idx]]
            else:
                cls2insts[cls].append(X[idx])
        cls2mean = {}
        for cls in cls2insts:
            cls2mean[cls] = numpy.mean(cls2insts[cls], axis=0).reshape(1, -1)
        return cls2mean


class LabelPerformance(object):
    """
    precision/recall/f1 calculation on TP/FN/FP values
    """
    def __init__(self, label):
        self._label = label
        self._tp = 0
        self._fn = 0
        self._fp = 0

    def increase_true_positive(self, k=1):
        self._tp += k

    def increase_false_negative(self, k=1):
        self._fn += k

    def increase_false_positive(self, k=1):
        self._fp += k

    @property
    def true_positive(self):
        return self._tp

    @property
    def false_negative(self):
        return self._fn

    @property
    def false_positive(self):
        return self._fp

    @property
    def precision(self):
        if self._tp + self._fp == 0:
            return -1
        else:
            return 1.0 * self._tp / (self._tp + self._fp)

    @property
    def recall(self):
        if self._tp + self._fn == 0:
            return -1
        else:
            return 1.0 * self._tp / (self._tp + self._fn)

    @property
    def f1(self):
        if self.precision == -1 or self.recall == -1 or self.precision == 0 or self.recall == 0:
            return -1
        else:
            return 2 / (1/self.precision + 1/self.recall)


class StrokeSettings(object):
    """
    json based configuration setting
    """
    def __init__(self, setting_file):
        self._file = setting_file
        self._setting = {}
        self.load()

    def load(self):
        self._setting = utils.load_json_data(self._file)

    @property
    def settings(self):
        return self._setting



def extract_doc_level_ann(ann_dump, output_folder):
    """

    extract doc level annotations and save to separate files
    :param ann_dump:
    :param output_folder:
    :return:
    """
    lines = utils.read_text_file(ann_dump)
    for l in lines:
        doc_ann = json.loads(l)
        utils.save_string(l, join(output_folder, doc_ann['docId'].split('.')[0] + '.json'))


def extract_all_doc_anns(dump_folder, output_folder):
    dumps = [f for f in listdir(dump_folder) if isfile(join(dump_folder, f))]
    for d in dumps:
        extract_doc_level_ann(join(dump_folder, d), output_folder)


def save_full_text(xml_file, output_dir):
    """
    recover full text from Informatics' xml format
    :param xml_file:
    :param output_dir:
    :return:
    """
    if not isfile(xml_file):
        return
    ed = EDIRDoc(xml_file)
    fn = basename(xml_file)
    name = fn.replace(r'-ann.xml', '.txt')
    logging.info('%s processed to be %s' % (fn, name))
    utils.save_string(ed.get_full_text, join(output_dir, name))


def process_files(read_dir, write_dir):
    utils.multi_thread_process_files(read_dir, file_extension='xml', num_threads=10,
                                     process_func=save_full_text, args=[write_dir])


def get_doc_level_inference(label_dir, ann_dir, file_key, type2insts, type2inst_2, t2missed):
    """
    learn concept to label inference from gold standard - i.e. querying SemEHR annotations to
    draw conclusions
    :param label_dir:
    :param ann_dir:
    :param file_key:
    :param type2insts:
    :param type2inst_2:
    :return:
    """
    label_file = '%s-ann.xml' % file_key
    ann_file = '%s.json' % file_key
    logging.info('working on %s' % join(label_dir, label_file))
    ed = EDIRDoc(join(label_dir, label_file))
    if not isfile(join(label_dir, label_file)):
        print 'not a file: %s' % join(label_dir, label_file)
        return
    sd = SemEHRAnnDoc(join(ann_dir, ann_file))
    sd.learn_mappings_from_labelled(ed, type2insts, t2missed)


def learn_concept_mappings(output_lst_folder):
    type2insts = {}
    type2insts_2 = {}
    label_dir = _gold_dir
    ann_dir = _ann_dir
    file_keys = [f.split('.')[0] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
    t2missed = {}
    for fk in file_keys:
        get_doc_level_inference(label_dir,
                                ann_dir,
                                fk,
                                type2insts,
                                type2insts_2,
                                t2missed)
    for t in type2insts:
        type2insts[t] = list(type2insts[t])
    logging.info(json.dumps(type2insts))

    s ='\n' * 2
    for t in type2insts_2:
        type2insts_2[t] = list(type2insts_2[t])
    s += json.dumps(type2insts_2)

    s += '\n' * 2
    labels = []
    defs = []
    for t in t2missed:
        t2missed[t] = list(set(t2missed[t]))
        utils.save_string('\n'.join(t2missed[t]) + '\n', join(output_lst_folder, t + '.lst'))
        labels += [l.lower() for l in t2missed[t]]
        defs.append(t + '.lst' + ':StrokeStudy:' + t)
    s += '\n' * 2
    s += '\n'.join(defs)
    s += json.dumps(t2missed)
    logging.info(s)


def learn_prediction_model(label, ann_dir=None, gold_dir=None, model_file=None, model_dir=None,
                           ml_model_file_ptn=None,
                           pca_dim=None,
                           pca_model_file=None,
                           max_dimension=None,
                           ignore_mappings=[],
                           viz_file=None, ignore_context=False, separate_by_label=False, full_text_dir=None):
    model_changed = False
    if model_file is not None:
        lm = LabelModel.deserialise(model_file)
    else:
        model_changed = True
        lm = LabelModel(label)
        lm.collect_tfidf_dimensions(ann_dir=ann_dir, gold_dir=gold_dir, ignore_context=ignore_context,
                                    separate_by_label=separate_by_label, full_text_dir=full_text_dir)
    lm.use_one_dimension_for_label = False
    lm.max_dimensions = max_dimension
    if ann_dir is not None:
        # bad_lables = lm.get_low_quality_labels(ann_dir, gold_dir)
        # logging.info(bad_lables)
        bad_lables = []
        data = lm.load_data(ann_dir, gold_dir, ignore_mappings=bad_lables, ignore_context=ignore_context,
                            separate_by_label=separate_by_label, ful_text_dir=full_text_dir)
        # if separate_by_label:
        for lbl in data['lbl2data']:
            X = data['lbl2data'][lbl]['X']
            Y = data['lbl2data'][lbl]['Y']
            n_true = 0
            for y in Y:
                if y == [1]:
                    n_true += 1
            if len(X) <= _min_sample_size:
                lm.add_rare_label(lbl, n_true * 1.0 / len(X))
                continue
            # ignore_mappings += data['bad_labels']
            lm.random_forest_learning(X, Y, output_file=ml_model_file_ptn % escape_lable_to_filename(lbl))
            # lm.svm_learning(X, Y, output_file=ml_model_file_ptn % escape_lable_to_filename(lbl))
            # lm.gaussian_nb(X, Y, output_file=ml_model_file_ptn % escape_lable_to_filename(lbl))
            logging.debug('%s, #insts: %s, #tps: %s' % (lbl, len(X), n_true))
            # if len(Y) > 20 and (.1< n_true * 1.0 / len(Y) < .9):
            #     correct_X = []
            #     incorrect_X = []
            #     correct_Y = []
            #     incorrect_Y = []
            #     for idx in xrange(len(Y)):
            #         if Y[idx] == [1]:
            #             correct_Y.append(Y[idx])
            #             correct_X.append(X[idx])
            #         else:
            #             incorrect_Y.append(Y[idx])
            #             incorrect_X.append(X[idx])
            #     bc = BinaryClusterClassifier(lbl)
            #     bc.cluster(correct_X, incorrect_X)
            #     lm.put_binary_cluster_classifier(lbl, bc)
                # LabelModel.cluster(correct_X, correct_Y)
                # LabelModel.cluster(incorrect_X, incorrect_Y)
                # logging.debug('doing KNN for %s' % lbl)
                # LabelModel.knn_classify(X, Y, output_file=ml_model_file_ptn % lbl)
            # lm.decision_tree_learning(X, Y, lm,
            #                           output_file=ml_model_file_ptn % escape_lable_to_filename(lbl),
            #                           pca_dim=pca_dim,
            #                           pca_file=pca_model_file,
            #                           # tree_viz_file=viz_file % escape_lable_to_filename(lbl),
            #                           lbl=lbl)

    if model_dir is not None and model_changed:
        lm.serialise(join(model_dir, '%s.lm' % label))
        logging.debug('%s.lm saved' % label)


def predict_label(model_file, test_ann_dir, test_gold_dir, ml_model_file_ptn, performance,
                  pca_model_file=None,
                  max_dimension=None,
                  ignore_mappings=[],
                  ignore_context=False,
                  separate_by_label=False,
                  full_text_dir=None,
                  file_pattern='%s-ann.xml',
                  id2conll=None,
                  label_whitelist=None):
    lm = LabelModel.deserialise(model_file)
    lm.max_dimensions = max_dimension
    data = lm.load_data(test_ann_dir, test_gold_dir, ignore_mappings=ignore_mappings, ignore_context=ignore_context,
                        separate_by_label=separate_by_label, verbose=False, ful_text_dir=full_text_dir)

    files = data['files']
    for d in files:
        if d not in id2conll:
            id2conll[d] = ConllDoc(join(test_gold_dir, file_pattern % d))
            if label_whitelist is not None:
                id2conll[d].set_label_white_list(label_whitelist)
    lbl2performances = {}
    for lbl in data['lbl2data']:
        this_performance = LabelPerformance(lbl)
        X = data['lbl2data'][lbl]['X']
        Y = data['lbl2data'][lbl]['Y']
        mtp = data['lbl2data'][lbl]['multiple_tps']
        doc_anns = data['lbl2data'][lbl]['doc_anns']
        if lbl in lm.rare_labels:
            logging.info('%s to be predicted using %s' % (lbl, lm.rare_labels[lbl]))
            LabelModel.predict_use_simple_stats(lm.rare_labels[lbl], Y, mtp,
                                                performance, separate_performance=this_performance,
                                                id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                                                doc_folder=test_gold_dir,
                                                label_whitelist=label_whitelist)
        else:
            if len(X) > 0:
                logging.debug('%s, dimensions %s' % (lbl, len(X[0])))
            bc = lm.get_binary_cluster_classifier(lbl)
            if bc is not None:
                complementary_classifiers = []
                for l in lm.cluster_classifier_dict:
                    if l != lbl:
                        complementary_classifiers.append(lm.cluster_classifier_dict[l])
                for idx in xrange(len(X)):
                    logging.debug('%s => %s' % (bc.classify(X[idx], complementary_classifiers=complementary_classifiers), Y[idx]))
            lm.predict_use_model(X, Y, 0, mtp, ml_model_file_ptn % escape_lable_to_filename(lbl), performance,
                                 pca_model_file=pca_model_file,
                                 separate_performance=this_performance,
                                 id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                                 doc_folder=test_gold_dir,
                                 label_whitelist=label_whitelist)
        lbl2performances[lbl] = this_performance
    CustomisedRecoginiser.print_performances(lbl2performances)
    logging.debug('missed instances: %s' % data['fns'])
    performance.increase_false_negative(data['fns'])


def escape_lable_to_filename(s):
    return s.replace('\\', '_').replace('/', '_')


def populate_semehr_results(label_dir, ann_dir, file_key,
                            label2performances, using_combined=False):
    label_file = '%s-ann.xml' % file_key
    ann_file = '%s.json' % file_key
    print join(label_dir, label_file)
    if not isfile(join(label_dir, label_file)):
        return

    ed = EDIRDoc(join(label_dir, label_file))
    cm = Concept2Mapping(_concept_mapping)
    cr = CustomisedRecoginiser(join(ann_dir, ann_file), cm)
    if using_combined:
        cr.validate_combined_performance(ed.get_ess_entities(), label2performances)
    else:
        cr.validate_mapped_performance(ed.get_ess_entities(), label2performances)


def populate_validation_results():
    label_dir = _gold_dir
    ann_dir = _ann_dir

    label2performances = {}
    file_keys = [f.split('.')[0] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
    for fk in file_keys:
        populate_semehr_results(label_dir, ann_dir, fk, label2performances, using_combined=False)
    CustomisedRecoginiser.print_performances(label2performances)


def do_learn_exp(viz_file, num_dimensions=[20], ignore_context=False, separate_by_label=False, conll_output_file=None):
    results = {}
    id2conll = {}
    for lbl in _labels:
        logging.info('working on [%s]' % lbl)
        _learning_model_file = _learning_model_dir + '/%s.lm' % lbl
        _ml_model_file_ptn = _learning_model_dir + '/' + lbl + '_%s_DT.model'
        _pca_model_file = None # '/afs/inf.ed.ac.uk/group/project/biomedTM/users/hwu/learning_models/%s_pca.model' % lbl
        pca_dim = None
        max_dimensions = num_dimensions

        t = lbl.replace('neg_', '')
        ignore_mappings = _ignore_mappings[t] if t in _ignore_mappings else []
        # remove previous model files
        logging.debug('removing previously learnt models...')
        for f in [f for f in listdir(_learning_model_dir) if isfile(join(_learning_model_dir, f)) and f.endswith('.model')]:
            remove(join(_learning_model_dir, f))
        for dim in max_dimensions:
            logging.info('dimension setting: %s' % dim)
            learn_prediction_model(lbl,
                                   ann_dir=_ann_dir,
                                   gold_dir=_gold_dir,
                                   ml_model_file_ptn=_ml_model_file_ptn,
                                   model_dir=_learning_model_dir,
                                   pca_dim=pca_dim,
                                   pca_model_file=_pca_model_file,
                                   max_dimension=dim,
                                   ignore_mappings=ignore_mappings,
                                   viz_file=viz_file,
                                   ignore_context=ignore_context,
                                   separate_by_label=separate_by_label,
                                   full_text_dir=_gold_text_dir)
            logging.debug('bad labels: %s' % ignore_mappings)
            pl = '%s dim[%s]' % (lbl, dim)
            performance = LabelPerformance(pl)
            results[pl] = performance
            predict_label(_learning_model_file,
                          _test_ann_dir,
                          _test_gold_dir,
                          _ml_model_file_ptn,
                          performance,
                          pca_model_file=_pca_model_file,
                          max_dimension=dim,
                          ignore_mappings=ignore_mappings,
                          ignore_context=ignore_context,
                          separate_by_label=separate_by_label,
                          full_text_dir=_test_text_dir,
                          id2conll=id2conll,
                          label_whitelist=_labels)
        CustomisedRecoginiser.print_performances(results)
    conll_output = ''
    for id in id2conll:
        doc_output = id2conll[id].conll_output
        conll_output += doc_output + '\n'
        logging.info('doc [%s]' % id)
        logging.info(doc_output)

    logging.info('total processed %s docs' % len(id2conll))
    if conll_output_file is not None:
        utils.save_string(conll_output, conll_output_file)
        logging.info('conll_output saved to [%s]' % conll_output_file)


def save_text_files(xml_dir, text_dr):
    process_files(xml_dir, text_dr)


def extact_doc_anns(semoutput_dir, doc_ann_dir):
    extract_all_doc_anns(semoutput_dir,
                         doc_ann_dir)


def merge_mappings_dictionary(map_files, dict_dirs, new_map_file, new_dict_folder):
    maps = [utils.load_json_data(mf) for mf in map_files]
    new_m = {}
    for m in maps:
        new_m.update(m)
    t2list = {}
    for dd in dict_dirs:
        lst_files = [f for f in listdir(dd) if isfile(join(dd, f)) and f.endswith('.lst')]
        for f in lst_files:
            t = f[:f.index('.')]
            labels = utils.read_text_file(join(dd, f))
            if t not in t2list:
                t2list[t] = set()
            for l in labels:
                if len(l) > 0:
                    t2list[t].add(l)
    utils.save_json_array(new_m, new_map_file)
    logging.info('mapping saved to %s' % new_map_file)
    for t in t2list:
        utils.save_string('\n'.join(list(t2list[t])) + '\n', join(new_dict_folder, t + '.lst'))
        logging.info('%s.lst saved' % t)
    logging.info('all done')


if __name__ == "__main__":
    logging.basicConfig(level='INFO', format='[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s')
    ss = StrokeSettings('./settings/tayside_annotator1.json')
    settings = ss.settings
    _min_sample_size = settings['min_sample_size']
    _ann_dir = settings['ann_dir']
    _gold_dir = settings['gold_dir']
    _test_ann_dir = settings['test_ann_dir']
    _test_gold_dir = settings['test_gold_dir']
    _gold_text_dir = settings['dev_full_text_dir']
    _test_text_dir = settings['test_fulltext_dir']
    _concept_mapping = settings['concept_mapping_file']
    _learning_model_dir = settings['learning_model_dir']
    _labels = utils.read_text_file(settings['entity_types_file'])
    _ignore_mappings = utils.load_json_data(settings['ignore_mapping_file'])
    _cm_obj = Concept2Mapping(_concept_mapping)
    # _cm_obj.load_gaz_dir(settings['concept_gaz_dir'])

    # 0. merging mapping & dictionaries
    # merge_mappings_dictionary(['/afs/inf.ed.ac.uk/group/project/biomedTM/users/hwu/tayside_concept_mapping.json',
    #                            '/afs/inf.ed.ac.uk/group/project/biomedTM/users/hwu/concept_mapping.json'],
    #                           ['/Users/honghan.wu/Documents/working/SemEHR-Working/toolkits/bio-yodie-1-2-1/finalize/tayside_gazetteer',
    #                            '/Users/honghan.wu/Documents/working/SemEHR-Working/toolkits/bio-yodie-1-2-1/finalize/ess_gazetteer'],
    #                           '/afs/inf.ed.ac.uk/group/project/biomedTM/users/hwu/merged_concept_mapping.json',
    #                           '/Users/honghan.wu/Documents/working/SemEHR-Working/toolkits/bio-yodie-1-2-1/finalize/merged_gzetteer')

    # 1. extract text files for annotation
    # save_text_files(settings['gold_dir'], settings['dev_full_text_dir'])
    # 2. run SemEHR on the text files
    # 3. extract doc anns into separate files from dumped JSON files
    # extact_doc_anns(settings['test_semehr_output_dir'],
    #                 settings['test_ann_dir'])
    # 4. learn umls concept to phenotype mappping
    # learn_concept_mappings(settings['gazetteer_dir'])
    # 5. learn phenotype inference
    do_learn_exp(settings['viz_file'],
                 num_dimensions=[30],
                 ignore_context=settings['ignore_context'] if 'ignore_context' in settings else False,
                 separate_by_label=True,
                 conll_output_file=settings['conll_output_file'])
