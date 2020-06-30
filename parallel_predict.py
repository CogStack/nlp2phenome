from annotation_docs import CustomisedRecoginiser, Concept2Mapping
import logging
from os.path import exists
from LabelModel import LabelModel
from predict_helper import label_model_predict
import mention_pattern as mp
import sqldbutils as du
import utils
import json


class ModelFactory(object):
    def __init__(self, phenotypes, model_dir):
        self._phenotypes = phenotypes
        self._learning_model_dir = model_dir
        self._no_model_labels = []
        self._phenotype2model = {}
        self._phenotype2model_file_pattern = {}
        self.load_models()

    def load_models(self):
        for phenotype in self._phenotypes:
            logging.info('working on [%s]' % phenotype)
            _learning_model_file = self._learning_model_dir + '/%s.lm' % phenotype

            if not exists(_learning_model_file):
                # if previous learnt model not exists, skip
                self._no_model_labels.append(phenotype)
                continue

            self._phenotype2model_file_pattern[phenotype] = self._learning_model_dir + '/' + phenotype + '_%s_DT.model'

            lm = LabelModel.deserialise(_learning_model_file)
            lm.max_dimensions = 30
            self._phenotype2model[phenotype] = lm

    def get_model_by_phenotype(self, phenotype):
        return self._phenotype2model[phenotype] if phenotype in self._phenotype2model else None

    def get_model_file_pattern(self, phenotype):
        return self._phenotype2model_file_pattern[phenotype] \
            if phenotype in self._phenotype2model_file_pattern else None

    @property
    def phenotypes(self):
        return self._phenotypes


def predict_doc_phenotypes(doc_key, doc_anns, doc_text, model_factory, ignore_mappings=[], mention_pattern=None):
    """
    load a document and do all phenotype predictions in one go
    this is designed for large amounts of documents to be loaded, for example, from databases
    :param doc_key:
    :param doc_anns:
    :param doc_text:
    :param model_factory:
    :param ignore_mappings:
    :param mention_pattern:
    :return:
    """
    cr = CustomisedRecoginiser(doc_key, ann_doc=doc_anns)
    cr.full_text = doc_text
    p2count = {}
    for p in model_factory:
        lm = model_factory.get_model_by_phenotype(p)
        if lm is None:
            logging.info('phenotype %s not found' % p)
            continue
        lbl2data = {}
        LabelModel.read_one_ann_doc(lm, cr, doc_key, lbl2data=lbl2data,
                                    ignore_mappings=ignore_mappings, ignore_context=True,
                                    separate_by_label=True)
        doc2predicted = {}
        label_model_predict(lm, model_factory.get_model_by_phenotype(p), lbl2data, doc2predicted,
                            mention_pattern=mention_pattern, mention_prediction_param=cr)
        p2count[p] = len(doc2predicted[doc_key])
    return p2count


def do_one_doc(doc_id, model_factory, mention_pattern, db_pool, sql_text_ptn, sql_ann_ptn, save_result_sql_ptn,
               update_doc_sql_ptn):
    container = []
    du.query_data(sql_ann_ptn % doc_id, pool=db_pool, container=container)
    if len(container) == 0:
        logging.info('%s anns not found' % doc_id)
        return
    doc_anns = json.loads(container[0])
    du.query_data(sql_text_ptn % doc_id, pool=db_pool, container=container)
    if len(container) == 0:
        logging.info('%s text not found' % doc_id)
        return
    text = container[0]
    p2count = predict_doc_phenotypes(doc_id, doc_anns, text, model_factory, mention_pattern=mention_pattern)
    du.query_data(save_result_sql_ptn % (doc_id, json.dumps(p2count)), container=None, pool=db_pool)
    du.query_data(update_doc_sql_ptn % doc_id, container=None, pool=db_pool)
    logging.info('%s done' % doc_id)


def run_parallel_prediction(settings):
    cm_obj = Concept2Mapping(settings['concept_mapping_file'])
    mp_inst = mp.MentionPattern(settings['pattern_folder'],
                                cm_obj.cui2label, in_action=True)
    db_pool = du.get_mysql_pooling(settings['dbconf'], num=10)
    doc_ids = []
    model_factory = ModelFactory(settings['phenotypes'], settings['model_dir'])
    du.query_data(settings['sql_docs4process'], pool=db_pool, container=doc_ids)
    utils.multi_thread_tasking(doc_ids, num_threads=settings['num_threads'], process_func=do_one_doc,
                               args=[model_factory, mp_inst, db_pool,
                                     settings['sql_text_ptn'],
                                     settings['sql_ann_ptn'],
                                     settings['save_result_sql_ptn'],
                                     settings['update_doc_sql_ptn']])
    logging.info('#docs: %s all done' % len(doc_ids))