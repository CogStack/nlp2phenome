# nlp2phenome
using AI models to infer patient phenotypes from identified named entities (instances of biomedical concepts)

## why
>Using natural language processing(NLP) to identify mentions of biomedical concepts from free text medical records is just the *first* step. There is often a gap between NLP results and what the clinical study is after. For example, a radiology report does not contain the term - `ischemic stroke`. Instead, it reports the patient had `blocked arteries` and `stroke`. To infer the "unspoken" `ischemic stroke`, a mechanism is needed to do such inferences from NLP identifiable mentions of `blocked arteries` and `stroke`. nlp2phenome is designed for doing this extra step from NLP to patient phenome.

## what
nlp2phenome was developed for a stroke subtyping study in Edinburgh University led by Dr Will Whitely. It is based on top of SemEHR results. It identified 2,922 mentions of 32 types of phenotypes from 266 radiology reports and achieved an average F1: 0.929; Precision: 0.925; Recall: 0.939. It uses various transparant machine learning models (e.g. decision tree, random forest) to learn the inference from NLP results to more insightful clinical phenotypes (such as subtypes of stroke). The image below is a decision tree learnt for negated tumour from radiology reports. Surprisingly, with specific feature selection methods, decision tree models outperforms the popular neural network based method. The other advantage is that the visualised decision trees can be verified or matched with clinical experts, or even compared to clinical guidelines. A working paper in progress, will update with a link soon.

![alt text](https://raw.githubusercontent.com/CogStack/nlp2phenome/master/neg-tumour-dt-learnt.png "decision tree learnt for negated tumour")

## data
two datasets (radiology reports) collected in Scotland
- [Edinburgh Stroke Study](http://www.dcn.ed.ac.uk/ess/) training data (364 reports), testing data (266 reports)
- Tayside radiology reports (300 reports)
