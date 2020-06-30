import logging
from os.path import isfile, join
from os import listdir
from annotation_docs import Concept2Mapping, CustomisedRecoginiser
from EDI_ann_doc import EDIRDoc, eHostGenedDoc
import joblib as jl


class LabelModel(object):
    """
    a machine learning based class for inferring phenotypes from NLP results
    features:
    - feature weighing
    - transparent models
    """
    def __init__(self, label, concept_mapping, max_dimensions=None):
        self._label = label
        self._concept_mapping = concept_mapping
        self._lbl_data = {}
        self._cui2label = {}
        self._selected_dims = None
        self._max_dimensions = 2000 if max_dimensions is None else max_dimensions
        self._tps = 0
        self._fps = 0
        self._lbl_one_dimension = True
        self._lbl2tfidf_dims = {}
        self._label_dimensions = []
        self._rare_labels = {}
        self._lbl2classifiers = {}

    @property
    def concept_mapping(self):
        return self._concept_mapping

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
        self.add_context_dimension(LabelModel.get_ann_dim_label(ann, generalise=True, no_negation=True), tp=tp, fp=fp,
                                   lbl=lbl)

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
            idf = 1.0 / ((1 if l in d['tps'] else 0) + (1 if l in d['fps'] else 0))
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
        self._lbl2tfidf_dims[lbl] = [(t[0], t[1] * 1.0 / max_score) for t in df[:k]]
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
        for l, score in self.get_top_tfidf_dimensions(self.max_dimensions, lbl=lbl):  # self.context_dimensions:
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
        cm = self.concept_mapping
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
                # collect cui labels
                for u in sanns['umls']:
                    self._cui2label[u.cui] = u.pref
                for c in context_anns:
                    self.add_context_dimension_by_annotation(c)

    def collect_tfidf_dimensions(self, ann_dir, gold_dir, ignore_context=False, separate_by_label=False,
                                 full_text_dir=None, eHostGD=False):
        cm = self.concept_mapping
        file_keys = [f[:f.rfind('.')] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
        # collect dimension labels
        tp_freq = 0
        fp_freq = 0
        label_type = self.label.replace('neg_', '')
        fn_freq = 0
        for fk in file_keys:
            cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), cm)
            fk = fk.replace('se_ann_', '')
            if full_text_dir is not None:
                cr.full_text_folder = full_text_dir
            if eHostGD:
                if not isfile(join(gold_dir, '%s.txt.knowtator.xml' % fk)):
                    continue
                gd = eHostGenedDoc(join(gold_dir, '%s.txt.knowtator.xml' % fk))
            else:
                if not isfile(join(gold_dir, '%s-ann.xml' % fk)):
                    continue
                gd = EDIRDoc(join(gold_dir, '%s-ann.xml' % fk))
            t = self.label.replace('neg_', '')
            anns = cr.get_anns_by_label(t)
            neg_anns = cr.get_anns_by_label('neg_' + t)

            # re-segement sentences
            # cr.re_segment_sentences(fk)
            # cr.relocate_all_anns(fk)
            # gd.relocate_anns(cr.get_full_text(fk))

            not_matched_gds = []
            for e in gd.get_ess_entities():
                if (ignore_context and e.label.replace('neg_', '') == label_type) \
                        or (not ignore_context and e.label == self.label):
                    not_matched_gds.append(e.id)
            for a in anns + neg_anns:
                # self.add_context_dimension_by_annotation(a)
                self.add_label_dimension_by_annotation(a)
                # if (not ignore_context) and ((a.negation != 'Negated' and self.label.startswith('neg_')) or \
                #         (a.negation == 'Negated' and not self.label.startswith('neg_'))):
                #     logging.info('skipped because context')
                #     continue

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

                sanns = cr.get_prior_anns(a, contenxt_depth=-1)
                context_anns = [] + sanns['umls'] + sanns['phenotype'] + cr.get_context_words(a, fk)
                # context_anns =  cr.get_context_words(a, fk)
                # collect cui labels
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
        cm = self.concept_mapping
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
        return sorted(lbls, key=lambda x: x[1])

    def load_data(self, ann_dir, gold_dir, verbose=True, ignore_mappings=[], ignore_context=False,
                  separate_by_label=False, ful_text_dir=None, eHostGD=False, annotated_anns={}):
        if ignore_context:
            logging.info('doing learning without considering contextual info')
        # print self.get_top_tfidf_dimensions(self.max_dimensions)
        cm = self.concept_mapping
        file_keys = [f[:f.rfind('.')] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
        lbl2data = {}
        false_negatives = 0
        lbl2tps = {}
        label_type = self.label.replace('neg_', '')
        query_label_perform = {}
        for fk in file_keys:
            cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), cm)
            fk = fk.replace('se_ann_', '')
            if ful_text_dir is not None:
                cr.full_text_folder = ful_text_dir
            if eHostGD:
                if not isfile(join(gold_dir, '%s.txt.knowtator.xml' % fk)):
                    continue
                # logging.debug('using GD file %s' % join(gold_dir, '%s.txt.knowtator.xml' % fk))
                gd = eHostGenedDoc(join(gold_dir, '%s.txt.knowtator.xml' % fk))
            else:
                if not isfile(join(gold_dir, '%s-ann.xml' % fk)):
                    continue
                logging.debug('using GD file %s' % join(gold_dir, '%s-ann.xml' % fk))
                gd = EDIRDoc(join(gold_dir, '%s-ann.xml' % fk))

            # re-segement sentences
            # cr.re_segment_sentences(fk)
            # cr.relocate_all_anns(fk)
            # gd.relocate_anns(cr.get_full_text(fk))

            not_matched_gds = []
            for e in gd.get_ess_entities():
                if (ignore_context and e.label.replace('neg_', '') == label_type) \
                        or (not ignore_context and e.label == self.label):
                    not_matched_gds.append(e.id)

            anns = cr.get_anns_by_label(self.label, ignore_mappings=ignore_mappings, no_context=ignore_context)
            if len(annotated_anns) > 0:
                if '%s.txt' % fk not in annotated_anns:
                    continue
                kept_anns = []
                for a in anns:
                    for aa in annotated_anns['%s.txt' % fk]:
                        if int(aa['s']) == a.start and int(aa['e']) == a.end:
                            kept_anns.append(a)
                anns = kept_anns
            for a in anns:
                logging.debug('%s, %s, %s' % (a.str, a.start, a.end))
                multiple_true_positives = 0
                t2anns = cr.get_prior_anns(a)
                # if len(t2anns['umls']) + len(t2anns['phenotype']) == 0:
                #     t2anns = cr.get_prior_anns(a, contenxt_depth=-2)
                context_anns = [] + t2anns['umls'] + t2anns['phenotype'] + \
                               cr.get_context_words(a, fk)
                # context_anns = cr.get_context_words(a, fk)
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
                                                    self.get_ann_dim_label(a) + ' // ' + ' | '.join(
                                                        self.get_ann_dim_label(a, generalise=True)
                                                        for a in context_anns), fk))

                lbl = LabelModel.get_label_specific_data(self, lbl2data, a, context_anns, fk, cr,
                                                         separate_by_label=separate_by_label)

                lbl2data[lbl]['multiple_tps'] += multiple_true_positives
                Y = lbl2data[lbl]['Y']
                Y.append([1 if matched else 0])
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
                    logging.debug('\t'.join(
                        ['M', g.str, str(g.negated), str(g.start), str(g.end), join(gold_dir, '%s-ann.xml' % fk)]))
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

    @staticmethod
    def get_label_specific_data(label_model, lbl2data, annotation, context_anns, fk, cr,
                                separate_by_label=False):
        a = annotation
        extra_dims = [1] if len(cr.get_containing_anns(a)) > 0 else [0]
        if separate_by_label:
            lbl = LabelModel.get_ann_query_label(a)
        else:
            lbl = 'united'
        if lbl not in lbl2data:
            lbl2data[lbl] = {'X': [], 'Y': [], 'multiple_tps': 0, 'doc_anns': []}
        X = lbl2data[lbl]['X']
        lbl2data[lbl]['doc_anns'].append({'d': fk, 'ann': a, 'label': label_model.label})
        X.append(label_model.encode_ann(a, context_anns, lbl=lbl, extra_dims=extra_dims))
        return lbl

    @staticmethod
    def read_one_ann_doc(label_model, cr, fk, lbl2data=None,
                         ignore_mappings=[], ignore_context=False, separate_by_label=False):
        if lbl2data is None:
            lbl2data = {}
        anns = cr.get_anns_by_label(label_model.label, ignore_mappings=ignore_mappings, no_context=ignore_context)
        for a in anns:
            t2anns = cr.get_prior_anns(a)
            context_anns = [] + t2anns['umls'] + t2anns['phenotype'] + cr.get_context_words(a, fk)
            # context_anns = cr.get_context_words(a, fk)
            LabelModel.get_label_specific_data(label_model, lbl2data, a, context_anns, fk, cr,
                                               separate_by_label=separate_by_label)
        return lbl2data

    def load_data_for_predict(self, ann_dir, ignore_mappings=[], ignore_context=False,
                              separate_by_label=False, full_text_dir=None):
        """
        load data for prediction - no ground truth exists
        :param ann_dir:
        :param ignore_mappings:
        :param ignore_context:
        :param separate_by_label:
        :param full_text_dir:
        :return:
        """
        if ignore_context:
            logging.info('doing learning without considering contextual info')

        cm = self.concept_mapping
        file_keys = [f[:f.rfind('.')] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
        lbl2data = {}
        for fk in file_keys:
            cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), cm)
            fk = fk.replace('se_ann_', '')
            if full_text_dir is not None:
                cr.full_text_folder = full_text_dir
            LabelModel.read_one_ann_doc(self, cr, fk, lbl2data=lbl2data,
                                        ignore_mappings=ignore_mappings, ignore_context=ignore_context,
                                        separate_by_label=separate_by_label)
        return {'lbl2data': lbl2data, 'files': file_keys}

    def serialise(self, output_file):
        jl.dump(self, output_file)

    @staticmethod
    def type_related_ann_filter(ann, cm_obj):
        if hasattr(ann, 'cui'):
            return not ann.cui.lower() in cm_obj.all_entities
            # return not ann.cui in _cm_obj.type2cocnepts(type)
        else:
            return not ann.str.lower() in cm_obj.all_entities
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
        if isinstance(ann, str):
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





