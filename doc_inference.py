import utils
import re
import json
import sys


class RuleConstruct(object):
    def __init__(self, phenotype):
        self._phenotype = phenotype
        self._negation = 'Affirmed'
        self._temporality = 'Recent'
        self._experiencer = 'Patient'

    @property
    def phenotype(self):
        return self._phenotype

    @phenotype.setter
    def phenotype(self, value):
        self._phenotype = value

    @property
    def negation(self):
        return self._negation

    @negation.setter
    def negation(self, value):
        self._negation = value

    @property
    def temporality(self):
        return self._temporality

    @temporality.setter
    def temporality(self, value):
        self._temporality = value

    @property
    def experiencer(self):
        return self._experiencer

    @experiencer.setter
    def experiencer(self, value):
        self._experiencer = value


class PhenotypeRule(object):
    def __init__(self):
        self._inclusion = []
        self._exclusion = []
        self._rule_label = None

    def inclusion_constructs(self):
        return self._inclusion

    def exclusion_units(self):
        return self._exclusion

    @property
    def rule_label(self):
        return self._rule_label

    @rule_label.setter
    def rule_label(self, value):
        self._rule_label = value

    @staticmethod
    def load_rules(rule_file):
        rules = utils.load_json_data(rule_file)
        prs = []
        for r in rules:
            pr = PhenotypeRule()
            pr.rule_label = r['label']
            prs.append(pr)
            pr.inclusion_constructs = [PhenotypeRule.get_rule_construct(c) for c in r['inclusions']]
            pr.exclusion_units = []
            for u in r['exclusion_units']:
                pr.exclusion_units.append([PhenotypeRule.get_rule_construct(c) for c in u])
        return prs

    @staticmethod
    def get_rule_construct(c):
        rc = RuleConstruct(c['phenotype'])
        if 'negation' in c:
            rc.negation = c['negation']
        if 'temporality' in c:
            rc.temporality = c['temporality']
        if 'experiencer' in c:
            rc.experiencer = c['experiencer']
        return rc


class PhenotypeRuleExecutor(object):
    def __init__(self):
        pass

    @staticmethod
    def apply_rules(doc_anns, rules):
        label_prov = []
        anns = [t['ann'] for t in doc_anns]
        for r in rules:
            prov = {"exclusion": [], "inclusion": None}
            label = ''
            inclusion_matched = PhenotypeRuleExecutor.match_rule_construct(r.inclusion_constructs, anns)
            if len(inclusion_matched) > 0:
                prov['inclusion'] = inclusion_matched
                for ec in r.exclusion_units:
                    exclusion_matched = PhenotypeRuleExecutor.match_rule_construct(ec, anns)
                    if len(exclusion_matched) > 0:
                        prov['exclusion'].append({'ec': ec, 'matched': exclusion_matched})
                if len(prov['exclusion']) == 0:
                    label = r.rule_label
            if label != '':  # or len(prov['exclusion']) > 0:
                label_prov.append({'label': label, 'prov': prov})
        return label_prov

    @staticmethod
    def match_ann_rule(rc, ann):
        return ann['minorType'] == rc.phenotype and ann['negation'] == rc.negation and ann[
            'temporality'] == rc.temporality and ann['experiencer'] == rc.experiencer

    @staticmethod
    def match_rule_construct(rc_list, anns):
        matched = []
        for ann in anns:
            m = True
            for rc in rc_list:
                if not PhenotypeRuleExecutor.match_ann_rule(rc, ann):
                    m = False
                    break
            if m:
                matched.append(ann)
        return matched


def load_patient_truth(truth_file):
    all_pids = []
    lines = utils.read_text_file(truth_file)
    type2ids = {}
    for l in lines:
        arr = l.split('\t')
        if arr[2] not in type2ids:
            type2ids[arr[2]] = []
        type2ids[arr[2]].append(arr[0])
        all_pids.append(arr[0])
    return type2ids, all_pids


def cal_performance(no_reports_pids, type2ids, doc_type2id, gd_labels, pred_label):
    gt_list = []
    for lbl in gd_labels:
        gt_list += type2ids[lbl]
    gt_ids = set(gt_list)
    pr_ids = set(doc_type2id[pred_label])
    print('\n*****%s******' % pred_label)

    false_negative = gt_ids - no_reports_pids - pr_ids
    false_positive = pr_ids - gt_ids
    print('total reported patients: %s, total truth: %s, predicted: %s, false negative:%s, false positive:%s'
          % (len(pids), len(gt_ids - no_reports_pids), len(pr_ids), len(false_negative), len(false_positive)))
    print('false negative: %s' % (false_negative))
    print('false positive: %s' % false_positive)


def doc_infer_with_ground_truth(patient_level_tsv, pids, doc_type2id):
    type2ids, all_pids = load_patient_truth(patient_level_tsv)
    no_reports_pids = set(all_pids) - set(pids)
    cal_performance(no_reports_pids, type2ids, doc_type2id, ['SAH', 'ICH'], 'primary haemorrhagic stroke')
    cal_performance(no_reports_pids, type2ids, doc_type2id, ['SAH'], 'subarachnoid haemorrhage')
    cal_performance(no_reports_pids, type2ids, doc_type2id, ['ICH'], 'intracerebra haemorrhage')
    cal_performance(no_reports_pids, type2ids, doc_type2id, ['Ischaemic'], 'ischaemic stroke')


def doc_infer(settings):
    rules = PhenotypeRule.load_rules(settings['rule_file'])
    d2predicted = utils.load_json_data(settings['doc_nlp_results'])
    doc_labels_output = settings['doc_inference_output']
    s = ''
    doc_type2id = {}
    pids = []
    for d in d2predicted:
        m = re.match(r'Stroke\_id\_(\d+)(\.\d+){0,1}', d)
        pid = d
        if m is not None:
            pid = m.group(1)
            pids.append(pid)
        label_provs = PhenotypeRuleExecutor.apply_rules(d2predicted[d], rules)
        print(pid, d, label_provs)
        for lp in label_provs:
            if lp['label'] != '':
                s += '%s\t%s\n' % (pid, lp['label'])
                if lp['label'] not in doc_type2id:
                    doc_type2id[lp['label']] = []
                doc_type2id[lp['label']].append(pid)

    pids = list(set(pids))
    print(json.dumps(pids))
    utils.save_string(s, doc_labels_output)
    if 'patient_level_truth_tsv' in settings:
        doc_infer_with_ground_truth(settings['patient_level_truth_tsv'], pids, doc_type2id)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('the syntax is [python doc_inference.py PROCESS_SETTINGS_FILE_PATH]')
    else:
        infer_settings = utils.load_json_data(sys.argv[1])
        doc_infer(infer_settings)
