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
import logging
from os.path import basename, isfile, join, split
from os import listdir, remove
import graphviz
import numpy


class PhenomeLearners(object):
    def __init__(self, setting):
        self._setting = setting

    @property
    def min_sample_size(self):
        return self._setting['min_sample_size']

    @staticmethod
    def decision_tree_learning(self, X, Y, lm, output_file=None, pca_dim=None, pca_file=None, tree_viz_file=None,
                               lbl='united', min_sample_size=25):
        if len(X) <= min_sample_size:
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
                                                          [(str(lm.cui2label[
                                                                    l.upper()]) + '(' + l.upper() + ')') if l.upper() in lm.cui2label else l
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
        for idx in range(len(dbm.labels_)):
            c = dbm.labels_[idx]
            cls = 'cls%s' % c
            if cls not in cls2label:
                cls2label[cls] = {'t': 0, 'f': 0}
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
    def cluster_predict(X, Y, fns, multiple_tps, model_file, performance,
                        separate_performance=None, min_sample_size=25):
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
        if all_true or len(X) <= min_sample_size:
            logging.warn('using querying instead of predicting')
            P = numpy.ones(len(X))
        else:
            logging.info('instance size %s' % len(P))
        for idx in range(len(P)):
            LabelPerformance.evaluate_to_performance(P[idx], Y[idx], [performance, separate_performance])

    @staticmethod
    def knn_classify(X, Y, output_file=None):
        knn = KNeighborsClassifier(n_neighbors=2).fit(X, Y)
        if output_file is not None:
            jl.dump(knn, output_file)
            logging.info('model file saved to %s' % output_file)

    @staticmethod
    def predict_use_simple_stats(tp_ratio, Y, multiple_tps, performance, ratio_cut_off=0.15, separate_performance=None,
                                 id2conll=None, doc_anns=None, file_pattern=None, doc_folder=None,
                                 label_whitelist=None, mp_predicted=None):
        P = numpy.ones(len(Y)) if tp_ratio >= ratio_cut_off else numpy.zeros(len(Y))
        P = PhenomeLearners.merge_with_pattern_prediction(P, mp_predicted)
        if multiple_tps > 0:
            performance.increase_true_positive(multiple_tps)
            if separate_performance is not None:
                separate_performance.increase_true_positive(multiple_tps)
        PhenomeLearners.cal_performance(P, Y, performance, separate_performance,
                                        id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                                        doc_folder=doc_folder,
                                        label_whitelist=label_whitelist)

    @staticmethod
    def merge_with_pattern_prediction(y_pred, mp_predict):
        if mp_predict is None:
            return y_pred
        y_merged = []
        print('>>>', y_pred, mp_predict)
        for idx in range(len(y_pred)):
            y_merged.append(y_pred[idx])
            if y_pred[idx] == 1 and mp_predict[idx] == 0:
                y_merged[idx] = 0
        return y_merged

    @staticmethod
    def predict_use_simple_stats_in_action(tp_ratio, item_size, ratio_cut_off=0.15,
                                           doc2predicted=None, doc_anns=None, mp_predicted=None):
        P = numpy.ones(item_size) if tp_ratio >= ratio_cut_off else numpy.zeros(item_size)
        P = PhenomeLearners.merge_with_pattern_prediction(P, mp_predicted)
        PhenomeLearners.collect_prediction(P, doc2predicted=doc2predicted, doc_anns=doc_anns)

    @staticmethod
    def cal_performance(P, Y, performance, separate_performance=None,
                        id2conll=None, doc_anns=None, file_pattern=None, doc_folder=None, label_whitelist=None):

        P = numpy.asarray(P).flatten().tolist()
        Y = numpy.asarray(Y).flatten().tolist()
        doc2predicted = {}
        for idx in range(len(P)):
            LabelPerformance.evaluate_to_performance(P[idx], Y[idx], [performance, separate_performance])
            if P[idx] == 1.0 and id2conll is not None and doc_anns is not None and doc_folder is not None:
                PhenomeLearners.collect_prediction(P, doc_anns, doc2predicted)
        # comment the following out to skip conll outputs
        # for d in doc2predicted:
        #     if d not in id2conll:
        #         id2conll[d] = ConllDoc(join(doc_folder, file_pattern % d))
        #         if label_whitelist is not None:
        #             id2conll[d].set_label_white_list(label_whitelist)
        #     cnll = id2conll[d]
        #     for anns in doc2predicted[d]:
        #         cnll.add_predicted_labels(anns)

    @staticmethod
    def predict_use_model(X, Y, fns, multiple_tps, model_file, performance,
                          pca_model_file=None, separate_performance=None,
                          id2conll=None, doc_anns=None, file_pattern=None, doc_folder=None,
                          label_whitelist=None, mp_predicted=None):
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
        if all_true:  # or len(X) <= _min_sample_size:
            logging.warning('using querying instead of predicting')
            P = numpy.ones(len(X))
        else:
            logging.info('instance size %s' % len(P))
        P = PhenomeLearners.merge_with_pattern_prediction(P, mp_predicted)
        PhenomeLearners.cal_performance(P, Y, performance, separate_performance,
                                        id2conll=id2conll, doc_anns=doc_anns, file_pattern=file_pattern,
                                        doc_folder=doc_folder, label_whitelist=label_whitelist)

    @staticmethod
    def predict_use_model_in_action(X, model_file, pca_model_file=None,
                                    doc2predicted=None, doc_anns=None, mp_predicted=None):
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

        if all_true:  # or len(X) <= _min_sample_size:
            logging.warning('using querying instead of predicting')
            P = numpy.ones(len(X))
        else:
            logging.info('instance size %s' % len(P))
        P = PhenomeLearners.merge_with_pattern_prediction(P, mp_predicted)
        PhenomeLearners.collect_prediction(P, doc2predicted=doc2predicted, doc_anns=doc_anns)

    @staticmethod
    def collect_prediction(P, doc_anns, doc2predicted):
        for idx in range(len(P)):
            if P[idx] == 1.0 and doc_anns is not None:
                d = doc_anns[idx]['d']
                labeled_ann = {'label': doc_anns[idx]['label'],
                               'ann': doc_anns[idx]['ann']}
                if d not in doc2predicted:
                    doc2predicted[d] = [labeled_ann]
                else:
                    doc2predicted[d].append(labeled_ann)


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
            return 2 / (1 / self.precision + 1 / self.recall)

    @staticmethod
    def evaluate_to_performance(predicted, labelled, performance_objects):
        if predicted == labelled:
            if predicted == 1.0:
                for pf in performance_objects:
                    if pf is not None:
                        pf.increase_true_positive()
        elif predicted == 1.0:
            for pf in performance_objects:
                if pf is not None:
                    pf.increase_false_positive()
        else:
            for pf in performance_objects:
                if pf is not None:
                    pf.increase_false_negative()

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
        for idx in range(len(dbm.labels_)):
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