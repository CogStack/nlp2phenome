from annotation_docs import CustomisedRecoginiser, Concept2Mapping
import logging
from os.path import exists
from LabelModel import LabelModel
from predict_helper import label_model_predict
import mention_pattern as mp
import sqldbutils as du
import utils
import json
import pandas as pd


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
            logging.info('loading on [%s]' % phenotype)
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

    def model_file_pattern(self, phenotype):
        return self._learning_model_dir + '/' + phenotype + '_%s_DT.model'


def predict_doc_phenotypes(doc_key, doc_anns, doc_text, model_factory, concept_mapping,
                           ignore_mappings=[], mention_pattern=None):
    """
    load a document and do all phenotype predictions in one go
    this is designed for large amounts of documents to be loaded, for example, from databases
    :param doc_key:
    :param doc_anns:
    :param doc_text:
    :param model_factory:
    :param concept_mapping:
    :param ignore_mappings:
    :param mention_pattern:
    :return:
    """
    cr = CustomisedRecoginiser(doc_key, concept_mapping=concept_mapping, ann_doc=doc_anns)
    cr.full_text = doc_text
    p2count = {}
    total = 0
    for p in model_factory.phenotypes:
        lm = model_factory.get_model_by_phenotype(p)
        if lm is None:
            logging.info('phenotype %s not found' % p)
            continue
        lbl2data = {}
        LabelModel.read_one_ann_doc(lm, cr, doc_key, lbl2data=lbl2data,
                                    ignore_mappings=ignore_mappings, ignore_context=True,
                                    separate_by_label=True)
        doc2predicted = {}
        label_model_predict(lm, model_factory.model_file_pattern(p), lbl2data, doc2predicted,
                            mention_pattern=mention_pattern, mention_prediction_param=cr)
        if doc_key in doc2predicted:
            p2count[p] = len(doc2predicted[doc_key])
            total += 1
    return p2count if total > 0 else None


def do_one_doc(doc_id, model_factory, concept_mapping, mention_pattern,
               db_pool, sql_text_ptn, sql_ann_ptn, save_result_sql_ptn,
               update_doc_sql_ptn):
    container = []
    du.query_data(sql_ann_ptn.format(**doc_id), pool=db_pool, container=container)
    if len(container) == 0:
        logging.info('%s anns not found' % doc_id)
        return
    doc_anns = json.loads(container[0]['anns'])
    patient_id = container[0]['patient_id']

    container = []
    du.query_data(sql_text_ptn.format(**doc_id), pool=db_pool, container=container)
    if len(container) == 0:
        logging.info('%s text not found' % doc_id)
        return
    text = container[0]['doc_content']
    container = []

    p2count = predict_doc_phenotypes(str(doc_id), doc_anns, text, model_factory, concept_mapping,
                                     mention_pattern=mention_pattern)

    if p2count is not None:
        save_dict = doc_id.copy()
        save_dict['result'] = json.dumps(p2count)
        save_dict['patient_id'] = patient_id
        du.query_data(save_result_sql_ptn.format(**save_dict), container=None, pool=db_pool)
        du.query_data(update_doc_sql_ptn.format(**doc_id), container=None, pool=db_pool)
    else:
        du.query_data(update_doc_sql_ptn.format(**doc_id), container=None, pool=db_pool)
        logging.info('%s empty phenotypes' % doc_id)
    logging.info('%s done' % doc_id)


def run_parallel_prediction(settings):
    cm_obj = Concept2Mapping(settings['concept_mapping_file'])
    mp_inst = mp.MentionPattern(settings['pattern_folder'],
                                cm_obj.cui2label, in_action=True)
    mp_inst = None
    db_pool = du.get_mysql_pooling(settings['dbconf'], num=30)
    doc_ids = []
    model_factory = ModelFactory(settings['phenotypes'], settings['model_dir'])
    du.query_data(settings['sql_docs4process'], pool=db_pool, container=doc_ids)
    # for d in doc_ids:
    #     do_one_doc(d, model_factory, cm_obj, mp_inst, db_pool,
    #                                  settings['sql_text_ptn'],
    #                                  settings['sql_ann_ptn'],
    #                                  settings['save_result_sql_ptn'],
    #                                  settings['update_doc_sql_ptn'])
    utils.multi_thread_tasking(doc_ids, num_threads=settings['num_threads'], process_func=do_one_doc,
                               args=[model_factory, cm_obj, mp_inst, db_pool,
                                     settings['sql_text_ptn'],
                                     settings['sql_ann_ptn'],
                                     settings['save_result_sql_ptn'],
                                     settings['update_doc_sql_ptn']])
    logging.info('#docs: %s all done' % len(doc_ids))


def initial_morbidity_row(phenotypes):
    r = {}
    for p in phenotypes:
        r[p] = 0
    return r


def add_data_row(data, patient, cur_row, phenotypes):
    if patient is None:
        return
    if 'patient_id' not in data:
        data['patient_id'] = []
    data['patient_id'].append(patient)
    for p in phenotypes:
        if p not in data:
            data[p] = []
        data[p].append(cur_row[p])


def collect_patient_morbidity_result(phenotypes, sql_result, db_conf, output_file):
    columns = ['patient_id'] + phenotypes
    # 1. query all results, assuming the result size is not too BIG (<1m) and ordered by patient_id
    db_pool = du.get_mysql_pooling(db_conf, num=10)
    container = []
    du.query_data(sql_result, pool=db_pool, container=container)
    data = {}
    cur_p = None
    cur_row = initial_morbidity_row(phenotypes)
    for d in container:
        if d['patient_id'] != cur_p:
            add_data_row(data, cur_p, cur_row, phenotypes)
            cur_p = d['patient_id']
            cur_row = initial_morbidity_row(phenotypes)
            logging.info('working on [%s]...' % cur_p)
        doc_result = json.loads(d['result'])
        for p in doc_result:
            cur_row[p] += doc_result[p]
    add_data_row(data, cur_p, cur_row, phenotypes)
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    logging.info('data saved to %s' % output_file)


if __name__ == "__main__":
    log_level = 'DEBUG'
    log_format = '[%(filename)s:%(lineno)d] %(name)s %(asctime)s %(message)s'
    logging.basicConfig(level='DEBUG', format=log_format)
    log_file = './settings/learning_process.log'
    logging.basicConfig(level=log_level, format=log_format, filename=log_file, filemode='w')
    # config = './settings/parallel_config.json'
    # run_parallel_prediction(utils.load_json_data(config))
    settings = utils.load_json_data('./settings/collect_morbidity.json')
    collect_patient_morbidity_result(settings['phenotypes'], settings['sql_result'],
                                     settings['db_conf'], settings['output_file'])
