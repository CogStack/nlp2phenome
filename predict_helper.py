from nlp_to_phenome import StrokeSettings, Concept2Mapping, escape_lable_to_filename
from LabelModel import LabelModel, CustomisedRecoginiser
from annotation_docs import PhenotypeAnn
from learners import PhenomeLearners
import utils
import logging
from os.path import join
from ann_converter import AnnConverter
from os import listdir
from os.path import isfile, exists
import sys


def predict(settings):
    ann_dir = settings['test_ann_dir']
    test_text_dir = settings['test_fulltext_dir']
    _concept_mapping = settings['concept_mapping_file']
    _learning_model_dir = settings['learning_model_dir']
    _labels = utils.read_text_file(settings['entity_types_file'])
    ignore_mappings = utils.load_json_data(settings['ignore_mapping_file'])
    _cm_obj = Concept2Mapping(_concept_mapping)

    doc2predicted = {}
    no_models_labels = []
    for phenotype in _labels:
        logging.info('working on [%s]' % phenotype)
        _learning_model_file = _learning_model_dir + '/%s.lm' % phenotype

        if not exists(_learning_model_file):
            # if previous learnt model not exists, skip
            no_models_labels.append(phenotype)
            continue
        
        _ml_model_file_ptn = _learning_model_dir + '/' + phenotype + '_%s_DT.model'

        lm = LabelModel.deserialise(_learning_model_file)
        lm.max_dimensions = 30
        data = lm.load_data_for_predict(
            ann_dir=ann_dir,
            concept_mapping_file=_concept_mapping,
            ignore_mappings=ignore_mappings, ignore_context=True,
            separate_by_label=True,
            full_text_dir=test_text_dir)
        for lbl in data['lbl2data']:
            X = data['lbl2data'][lbl]['X']
            logging.debug(X)
            doc_anns = data['lbl2data'][lbl]['doc_anns']
            label_model_predict(lm, _ml_model_file_ptn, data['lbl2data'], doc2predicted)
    return doc2predicted, no_models_labels


def label_model_predict(lm, model_file_pattern, lbl2data, doc2predicted,
                        mention_pattern=None, mention_prediction_param=None):
    for lbl in lbl2data:
        mp_predicted = None
        if mention_pattern is not None:
            mp_predicted = mention_pattern.predict(lbl2data[lbl]['doc_anns'], cr=mention_prediction_param)
        X = lbl2data[lbl]['X']
        doc_anns = lbl2data[lbl]['doc_anns']
        if lbl in lm.rare_labels:
            logging.info('%s to be predicted using %s' % (lbl, lm.rare_labels[lbl]))
            PhenomeLearners.predict_use_simple_stats_in_action(lm.rare_labels[lbl],
                                                               item_size=len(X),
                                                               doc2predicted=doc2predicted,
                                                               doc_anns=doc_anns,
                                                               mp_predicted=mp_predicted)
        else:
            if len(X) > 0:
                logging.debug('%s, dimensions %s' % (lbl, len(X[0])))
            PhenomeLearners.predict_use_model_in_action(X, model_file=model_file_pattern % escape_lable_to_filename(lbl),
                                                        pca_model_file=None,
                                                        doc2predicted=doc2predicted,
                                                        doc_anns=doc_anns,
                                                        mp_predicted=mp_predicted)


def hybrid_prediciton(settings):
    d2p, labels2work = predict(settings)
    ann_dir = settings['test_ann_dir']
    test_text_dir = settings['test_fulltext_dir']
    _concept_mapping = settings['concept_mapping_file']
    _learning_model_dir = settings['learning_model_dir']
    _labels = utils.read_text_file(settings['entity_types_file'])
    ignore_mappings = utils.load_json_data(settings['ignore_mapping_file'])
    _cm_obj = Concept2Mapping(_concept_mapping)
    file_keys = [f[:f.rfind('.')].replace('se_ann_', '') for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
    logging.info('labels to use direct nlp prediction: [%s]' % labels2work)
    
    # convert SemEHRAnn to PhenotypeAnn
    doc2predicted = {}
    for d in d2p:
        for t in d2p[d]:
            ann = t['ann']
            if hasattr(ann, 'cui'):
                lbl = _cm_obj.concept2label[ann.cui][0]
                pheAnn = PhenotypeAnn(ann.str, ann.start, ann.end, ann.negation, ann.temporality, ann.experiencer,
                                          'StudyName', lbl)
                put_ann_label(lbl, pheAnn, doc2predicted, d)
            else:
                put_ann_label(ann.minor_type, ann, doc2predicted, d)
    for fk in file_keys:
        cr = CustomisedRecoginiser(join(ann_dir, 'se_ann_%s.json' % fk), _concept_mapping)
        d = fk
        for ann in cr.annotations:
            if ann.cui in _cm_obj.concept2label:
                lbl = _cm_obj.concept2label[ann.cui][0]
                if lbl in labels2work:
                    pheAnn = PhenotypeAnn(ann.str, ann.start, ann.end, ann.negation, ann.temporality, ann.experiencer,
                                          'StudyName', lbl)
                    put_ann_label(lbl, pheAnn, doc2predicted, d)
        for ann in cr.phenotypes:
            if ann.minor_type in labels2work:
                put_ann_label(ann.minor_type, ann, doc2predicted, d)
    return doc2predicted


def direct_nlp_prediction(settings):
    ann_dir = settings['test_ann_dir']
    test_text_dir = settings['test_fulltext_dir']
    _concept_mapping = settings['concept_mapping_file']
    _learning_model_dir = settings['learning_model_dir']
    _labels = utils.read_text_file(settings['entity_types_file'])
    ignore_mappings = utils.load_json_data(settings['ignore_mapping_file'])
    _cm_obj = Concept2Mapping(_concept_mapping)
    file_keys = [f[:f.rfind('.')].replace('se_ann_', '') for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
    doc2predicted = {}
    for fk in file_keys:
        cr = CustomisedRecoginiser(join(ann_dir, 'se_ann_%s.json' % fk), _concept_mapping)
        d = fk
        for ann in cr.annotations:
            if ann.cui in _cm_obj.concept2label:
                lbl = _cm_obj.concept2label[ann.cui][0]
                pheAnn = PhenotypeAnn(ann.str, ann.start, ann.end, ann.negation, ann.temporality, ann.experiencer,
                                      'StudyName', lbl)
                put_ann_label(lbl, pheAnn, doc2predicted, d)
        for ann in cr.phenotypes:
            put_ann_label(ann.minor_type, ann, doc2predicted, d)
    return doc2predicted


def put_ann_label(lbl, pheAnn, doc2predicted, d):
    labeled_ann = {'label': lbl,
                   'ann': pheAnn}
    if d not in doc2predicted:
        doc2predicted[d] = [labeled_ann]
    else:
        doc2predicted[d].append(labeled_ann)


def output_eHOST_format(doc2precited, output_folder):
    for d in doc2precited:
        xml = AnnConverter.to_eHOST(d, doc2precited[d])
        utils.save_string(xml, join(output_folder, '%s.txt.knowtator.xml' % d))


def predict_to_eHOST_results(predict_setting):
    ss = StrokeSettings(predict_setting)
    if 'predict_mode' in ss.settings and ss.settings['predict_mode'] == 'direct_nlp':
        logging.info('predicting with direct nlp...')
        predicted_results = direct_nlp_prediction(ss.settings)
    elif 'predict_mode' in ss.settings and ss.settings['predict_mode'] == 'hybrid':
        predicted_results = hybrid_prediciton(ss.settings)
    else:
        logging.info('predicting...')
        predicted_results = predict(ss.settings)
    output_eHOST_format(predicted_results, ss.settings['output_folder'])
    logging.info('results saved to %s' % ss.settings['output_folder'])
    if 'output_file' in ss.settings:
        d2ann = {}
        for d in predicted_results:
            d2ann[d] = [{'label': t['label'], 'ann': t['ann'].to_dict()} for t in predicted_results[d]]
        utils.save_json_array(d2ann, ss.settings['output_file'])


if __name__ == "__main__":
    logging.basicConfig(level='DEBUG', format='[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s')
    # predict_to_eHOST_results('./settings/prediction_task_direct.json')
    if len(sys.argv) != 2:
        print('the syntax is [python prediction_helper.py PROCESS_SETTINGS_FILE_PATH]')
    else:
        predict_to_eHOST_results(sys.argv[1])