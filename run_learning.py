from sklearn.model_selection import KFold
from os import listdir, makedirs
from os.path import isfile, join, isdir
import shutil
from nlp_to_phenome import run_learning
import utils
import logging


def run_kfold_learning(k, corpus_folder, semehr_folder, gold_folder, working_folder):
    kf = KFold(n_splits=k)
    files = [f for f in listdir(corpus_folder) if isfile(join(corpus_folder, f))]
    k = 0
    for train_idx, test_idx in kf.split(files):
        reset_folder(working_folder)
        # copy files
        train_ann_dir = join(working_folder, 'ann')
        train_gold_dir = join(working_folder, 'gold')    
        train_text_dir = join(working_folder, 'train_corpus')
        test_ann_dir = join(working_folder, 'test_ann')
        test_gold_dir = join(working_folder, 'test_gold')
        test_text_dir = join(working_folder, 'test_corpus')
        for idx in train_idx:
            shutil.copy(join(corpus_folder, files[idx]), join(train_text_dir, files[idx]))
            ann_file = 'se_ann_%s.json' % files[idx].replace('.txt', '')            
            gold_file = '%s.knowtator.xml' % files[idx]
            shutil.copy(join(semehr_folder, ann_file), join(train_ann_dir, ann_file))
            shutil.copy(join(gold_folder, gold_file), join(train_gold_dir, gold_file))
        for idx in test_idx:
            shutil.copy(join(corpus_folder, files[idx]), join(test_text_dir, files[idx]))
            ann_file = 'se_ann_%s.json' % files[idx].replace('.txt', '')
            gold_file = '%s.knowtator.xml' % files[idx]
            shutil.copy(join(semehr_folder, ann_file), join(test_ann_dir, ann_file))
            shutil.copy(join(gold_folder, gold_file), join(test_gold_dir, gold_file))
        performance = run_learning()
        utils.save_string(performance, join(working_folder, 'folder_%s_perf.tsv' % k))
        k += 1
        logging.info('round %s done' % k)


def reset_folder(working_folder):
    # clear working folder
    for d in listdir(working_folder):
        if isdir(join(working_folder, d)):
            shutil.rmtree(join(working_folder, d))

    train_ann_dir = join(working_folder, 'ann')
    train_gold_dir = join(working_folder, 'gold')    
    train_text_dir = join(working_folder, 'train_corpus')
    test_ann_dir = join(working_folder, 'test_ann')
    test_gold_dir = join(working_folder, 'test_gold')
    test_text_dir = join(working_folder, 'test_corpus')
    learning_model_dir = join(working_folder, 'models')
    makedirs(train_ann_dir)
    makedirs(train_gold_dir)    
    makedirs(train_text_dir)
    makedirs(test_ann_dir)
    makedirs(test_gold_dir)
    makedirs(test_text_dir)
    makedirs(learning_model_dir)


def run_it():
    corpus_folder = '/data/annotated_data/corpus'
    gold_folder = '/data/annotated_data/gold'
    semehr_folder = '/data/semehr_results'
    working_folder = '/data/learning'
    run_kfold_learning(10, corpus_folder, semehr_folder, gold_folder, working_folder)
    

if __name__ == "__main__":
    run_it()