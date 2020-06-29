#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nlp2phenome
using AI models to infer patient phenotypes from identified named entities (instances of biomedical concepts)
"""
import utils
from os.path import basename, isfile, join
from os import listdir
import json
import logging
from LabelModel import LabelModel
import mention_pattern as mp
from annotation_docs import SemEHRAnnDoc, CustomisedRecoginiser, Concept2Mapping
from EDI_ann_doc import EDIRDoc, ConllDoc, eHostDoc
from learners import LabelPerformance, PhenomeLearners


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
        print('not a file: %s' % join(label_dir, label_file))
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

    s = '\n' * 2
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
                           viz_file=None, ignore_context=False, separate_by_label=False, full_text_dir=None,
                           eHostGD=False):
    model_changed = False
    if model_file is not None:
        lm = LabelModel.deserialise(model_file)
    else:
        model_changed = True
        lm = LabelModel(label, _cm_obj)
        lm.collect_tfidf_dimensions(ann_dir=ann_dir, gold_dir=gold_dir, ignore_context=ignore_context,
                                    separate_by_label=separate_by_label, full_text_dir=full_text_dir, eHostGD=eHostGD)
    lm.use_one_dimension_for_label = False
    lm.max_dimensions = max_dimension
    if ann_dir is not None:
        # bad_lables = lm.get_low_quality_labels(ann_dir, gold_dir)
        # logging.info(bad_lables)
        bad_lables = []
        data = lm.load_data(ann_dir, gold_dir, ignore_mappings=bad_lables, ignore_context=ignore_context,
                            separate_by_label=separate_by_label, ful_text_dir=full_text_dir, eHostGD=eHostGD)
        # if separate_by_label:
        for lbl in data['lbl2data']:
            X = data['lbl2data'][lbl]['X']
            Y = data['lbl2data'][lbl]['Y']
            n_true = 0
            for y in Y:
                if y == [1]:
                    n_true += 1
            logging.debug('training data: %s, dimensions %s, insts %s' % (lbl, len(X[0]), len(X)))
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
                  label_whitelist=None,
                  eHostGD=False, mention_pattern=None):
    lm = LabelModel.deserialise(model_file)
    lm.max_dimensions = max_dimension
    data = lm.load_data(test_ann_dir, test_gold_dir, ignore_mappings=ignore_mappings, ignore_context=ignore_context,
                        separate_by_label=separate_by_label, verbose=False, ful_text_dir=full_text_dir, eHostGD=eHostGD)

    files = data['files']
    for d in files:
        d = d.replace('se_ann_', '')
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
        mp_predicted = None
        if mention_pattern is not None:
            mp_predicted = mention_pattern.predict(doc_anns)
        if lbl in lm.rare_labels:
            logging.info('%s to be predicted using %s' % (lbl, lm.rare_labels[lbl]))
            PhenomeLearners.predict_use_simple_stats(
                lm.rare_labels[lbl], Y, mtp,
                performance, separate_performance=this_performance,
                id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                doc_folder=test_gold_dir,
                label_whitelist=label_whitelist, mp_predicted=mp_predicted
            )
        else:
            if len(X) > 0:
                logging.debug('predict data: %s, dimensions %s, insts %s' % (lbl, len(X[0]), len(X)))
            bc = lm.get_binary_cluster_classifier(lbl)
            if bc is not None:
                complementary_classifiers = []
                for l in lm.cluster_classifier_dict:
                    if l != lbl:
                        complementary_classifiers.append(lm.cluster_classifier_dict[l])
                for idx in range(len(X)):
                    logging.debug(
                        '%s => %s' % (bc.classify(X[idx], complementary_classifiers=complementary_classifiers), Y[idx]))
            lm.predict_use_model(X, Y, 0, mtp, ml_model_file_ptn % escape_lable_to_filename(lbl), performance,
                                 pca_model_file=pca_model_file,
                                 separate_performance=this_performance,
                                 id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                                 doc_folder=test_gold_dir,
                                 label_whitelist=label_whitelist, mp_predicted=mp_predicted)
        lbl2performances[lbl] = this_performance
    perform_str = CustomisedRecoginiser.print_performances(lbl2performances)
    logging.debug('missed instances: %s' % data['fns'])
    performance.increase_false_negative(data['fns'])
    return perform_str


def escape_lable_to_filename(s):
    return s.replace('\\', '_').replace('/', '_')


def populate_semehr_results(label_dir, ann_dir, file_key,
                            label2performances, using_combined=False):
    label_file = '%s-ann.xml' % file_key
    ann_file = '%s.json' % file_key
    print(join(label_dir, label_file))
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


def do_learn_exp(viz_file, num_dimensions=[20], ignore_context=False, separate_by_label=False, conll_output_file=None,
                 eHostGD=False, mention_pattern=None):
    results = {}
    id2conll = {}
    result_str = ''
    for lbl in _labels:
        logging.info('working on [%s]' % lbl)
        _learning_model_file = _learning_model_dir + '/%s.lm' % lbl
        _ml_model_file_ptn = _learning_model_dir + '/' + lbl + '_%s_DT.model'
        _pca_model_file = None
        pca_dim = None
        max_dimensions = num_dimensions

        t = lbl.replace('neg_', '')
        ignore_mappings = _ignore_mappings[t] if t in _ignore_mappings else []
        # remove previous model files logging.debug('removing previously learnt models...') for f in [f for f in
        # listdir(_learning_model_dir) if isfile(join(_learning_model_dir, f)) and f.endswith('.model')]: remove(
        # join(_learning_model_dir, f))
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
                                   full_text_dir=_gold_text_dir,
                                   eHostGD=eHostGD)
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
                          file_pattern=_gold_file_pattern,
                          id2conll=id2conll,
                          label_whitelist=_labels,
                          eHostGD=eHostGD, mention_pattern=mention_pattern)
        result_str = CustomisedRecoginiser.print_performances(results)
    # conll_output = ''
    # for id in id2conll:
    #     doc_output = id2conll[id].conll_output
    #     conll_output += doc_output + '\n'
    #     logging.info('doc [%s]' % id)
    #     logging.info(doc_output)
    #
    # logging.info('total processed %s docs' % len(id2conll))
    # if conll_output_file is not None:
    #     utils.save_string(conll_output, conll_output_file)
    #     logging.info('conll_output saved to [%s]' % conll_output_file)
    return result_str


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


def test_eHost_doc():
    d = eHostDoc('/Users/honghan.wu/Desktop/ehost_sample.xml')
    print([(e.label, e.start, e.end, e.str) for e in d.get_ess_entities()])


def run_learning():
    log_level = 'DEBUG'
    log_format = '[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s'
    logging.basicConfig(level='DEBUG', format=log_format)
    log_file = './settings/processing.log'
    logging.basicConfig(level=log_level, format=log_format)
    ss = StrokeSettings('./settings/settings.json')
    settings = ss.settings
    global _min_sample_size, _ann_dir, _gold_dir, _test_ann_dir, _test_gold_dir, _gold_text_dir, _test_text_dir, _concept_mapping, _learning_model_dir
    global _labels, _gold_file_pattern, _ignore_mappings, _eHostGD, _cm_obj
    global _annotated_anns
    _annotated_anns = {}
    if 'annotated_anns' in settings['annotated_anns_file']:
        _annotated_anns = utils.load_json_data(settings['annotated_anns_file'])
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
    _gold_file_pattern = "%s_ann.xml" if 'gold_file_pattern' not in settings else settings['gold_file_pattern']
    _ignore_mappings = utils.load_json_data(settings['ignore_mapping_file'])
    _eHostGD = settings['eHostGD'] if 'eHostGD' in settings else False
    _cm_obj = Concept2Mapping(_concept_mapping)

    mp_inst = mp.MentionPattern(settings['pattern_folder'], _cm_obj.cui2label,
                                csv_file=settings['csv_file'], ann_folder=_test_ann_dir)
    return do_learn_exp(settings['viz_file'],
                        num_dimensions=[50],
                        ignore_context=settings['ignore_context'] if 'ignore_context' in settings else False,
                        separate_by_label=True,
                        conll_output_file=settings['conll_output_file'], eHostGD=_eHostGD, mention_pattern=mp_inst)


if __name__ == "__main__":
    log_level = 'DEBUG'
    log_format = '[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s'
    logging.basicConfig(level='DEBUG', format=log_format)
    log_file = './settings/processing.log'
    logging.basicConfig(level=log_level, format=log_format)
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
