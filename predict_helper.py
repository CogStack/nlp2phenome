from nlp_to_phenome import LabelModel, LabelPerformance, StrokeSettings, Concept2Mapping, escape_lable_to_filename, \
    CustomisedRecoginiser, PhenotypeAnn
import utils
import logging
from os.path import join
from ann_converter import AnnConverter
from os import listdir
from os.path import isfile


def predict(settings):
    ann_dir = settings['test_ann_dir']
    test_text_dir = settings['test_fulltext_dir']
    _concept_mapping = settings['concept_mapping_file']
    _learning_model_dir = settings['learning_model_dir']
    _labels = utils.read_text_file(settings['entity_types_file'])
    ignore_mappings = utils.load_json_data(settings['ignore_mapping_file'])
    _cm_obj = Concept2Mapping(_concept_mapping)

    doc2predicted = {}
    for phenotype in _labels:
        logging.info('working on [%s]' % phenotype)
        _learning_model_file = _learning_model_dir + '/%s.lm' % phenotype
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
            if lbl in lm.rare_labels:
                logging.info('%s to be predicted using %s' % (lbl, lm.rare_labels[lbl]))
                LabelModel.predict_use_simple_stats_in_action(lm.rare_labels[lbl],
                                                              item_size=len(X),
                                                              doc2predicted=doc2predicted,
                                                              doc_anns=doc_anns)
            else:
                if len(X) > 0:
                    logging.debug('%s, dimensions %s' % (lbl, len(X[0])))
                lm.predict_use_model_in_action(X, model_file=_ml_model_file_ptn % escape_lable_to_filename(lbl),
                                               pca_model_file=None,
                                               doc2predicted=doc2predicted,
                                               doc_anns=doc_anns)
    return doc2predicted


def direct_nlp_prediction(settings):
    ann_dir = settings['test_ann_dir']
    test_text_dir = settings['test_fulltext_dir']
    _concept_mapping = settings['concept_mapping_file']
    _learning_model_dir = settings['learning_model_dir']
    _labels = utils.read_text_file(settings['entity_types_file'])
    ignore_mappings = utils.load_json_data(settings['ignore_mapping_file'])
    _cm_obj = Concept2Mapping(_concept_mapping)
    file_keys = [f.split('.')[0] for f in listdir(ann_dir) if isfile(join(ann_dir, f))]
    doc2predicted = {}
    for fk in file_keys:
        cr = CustomisedRecoginiser(join(ann_dir, '%s.json' % fk), _concept_mapping)
        d = cr.full_text_file_pattern % fk
        for ann in cr.annotations:
            if ann.cui in _concept_mapping.cui2label:
                lbl = _concept_mapping.cui2label[ann.cui]
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
    else:
        logging.info('predicting...')
        predicted_results = predict(ss.settings)
    output_eHOST_format(predicted_results, ss.settings['output_folder'])
    logging.info('results saved to %s' % ss.settings['output_folder'])


if __name__ == "__main__":
    logging.basicConfig(level='DEBUG', format='[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s')
    predict_to_eHOST_results('./settings/prediction_task.json')