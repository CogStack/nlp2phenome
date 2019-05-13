from nlp_to_phenome import LabelModel, LabelPerformance, StrokeSettings, Concept2Mapping, escape_lable_to_filename
import utils
import logging
from os.path import join
from ann_converter import AnnConverter


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


def output_eHOST_format(doc2precited, output_folder):
    for d in doc2precited:
        xml = AnnConverter.to_eHOST(d, doc2precited[d])
        utils.save_string(xml, join(output_folder, '%s.txt.knowtator.xml' % d))


def predict_to_eHOST_results(predict_setting):
    ss = StrokeSettings(predict_setting)
    logging.info('predicting...')
    output_eHOST_format(predict(ss.settings), ss.settings['output_folder'])
    logging.info('results saved to %s' % ss.settings['output_folder'])


if __name__ == "__main__":
    logging.basicConfig(level='DEBUG', format='[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s')
    predict_to_eHOST_results('./settings/prediction_task.json')