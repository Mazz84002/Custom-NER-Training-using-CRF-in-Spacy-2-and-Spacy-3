# Python script to evaluate performance of crf training (accuracy, f1-core, precisison, and speed)

# https://github.com/juand-r/entity-recognition-datasets/tree/master/data

import time

import spacy
nlp = spacy.load("en_core_web_md") # to import spacy tokenizer

# evaluate once with spacy 2 and once with spacy 3
'''
    results is a dictionary, with key as the dataset name and value as another dictionary consisting results
'''
results = {}


def spacy_version():
    ver = int(str(spacy.__version__)[0])
    return ver

def result_extractor():
    for item in crf_extractor.eval(test_examples):
        print(item)


train_sets = [
    "/Users/mazz/Documents/Programming/Python/Mini-Project/DataSet/conll2003 train large.txt",
    "/Users/mazz/Documents/Programming/Python/Mini-Project/DataSet/conll2003 train small.txt",
    "/Users/mazz/Documents/Programming/Python/Mini-Project/DataSet/dataset3_train.txt"
]

test_sets = [
    "/Users/mazz/Documents/Programming/Python/Mini-Project/DataSet/conll2003 test.txt",
    "/Users/mazz/Documents/Programming/Python/Mini-Project/DataSet/conll2003 test.txt",
    "/Users/mazz/Documents/Programming/Python/Mini-Project/DataSet/dataset3_test.txt"
]

# ------------------------------------------------------------------------------------------------------------------------------------------------------
for p in range(0, 3):
    print('----------------------------------- Dataset %s -----------------------------------' % (p+1))

    results = {}
    r = results

    from spacy_crfsuite import CRFExtractor
    from component_config import return_custom_components
    component_config = return_custom_components()
    crf_extractor = CRFExtractor(component_config=component_config)

    # reading contents
    from tqdm.notebook import tqdm_notebook
    from spacy_crfsuite import read_file
    from spacy_crfsuite.train import gold_example_to_crf_tokens
    from spacy_crfsuite.tokenizer import SpacyTokenizer

    def read_examples(file, tokenizer, use_dense_features=False, limit=None):
        examples = []
        it = read_file(file)
        it = it[:limit] if limit else it
        for raw_example in tqdm_notebook(it, desc=file):
            crf_example = gold_example_to_crf_tokens(
                raw_example, 
                tokenizer=tokenizer, 
                use_dense_features=use_dense_features, 
                bilou=False
            )
            examples.append(crf_example)
        return examples

    tokenizer = SpacyTokenizer(nlp)

    # -------------------- OPTIONAL ---------------------------
    dev_examples = None
    # dev_examples = read_examples("conll03/valid.conll", tokenizer, use_dense_features=use_dense_features)

    if dev_examples:
        rs = crf_extractor.fine_tune(dev_examples, cv=5, n_iter=30, random_state=42)
        print("best params:", rs.best_params_, ", score:", rs.best_score_)
        crf_extractor.component_config.update(rs.best_params_)
    # -------------------- OPTIONAL ---------------------------

    if spacy_version() == 3:
        print('spacy version 3')
        # training using dataset
        st = time.time()
        train_examples = read_examples(train_sets[p], tokenizer=tokenizer, use_dense_features=False)
        crf_extractor.train(train_examples, dev_samples=dev_examples)
        et = time.time()
        elapsed_time = et - st
        r['time(sec)'] = elapsed_time

        test_examples = read_examples(test_sets[p], tokenizer=tokenizer, use_dense_features=False)
        r['score'] = crf_extractor.eval(test_examples)[0]
        r['stats'] = crf_extractor.eval(test_examples)[1]

        import spacy
        from spacy.language import Language
        from spacy_crfsuite import CRFEntityExtractor

        @Language.factory("ner-crf-1") # give a name to the new factory (unique)
        def create_my_component(nlp, name):
            #crf_extractor = CRFExtractor().from_disk("path-to-model")
            pipe = CRFEntityExtractor(nlp, crf_extractor=crf_extractor) # convert crf model into spacy pipeline component
            return pipe

        nlp = spacy.load("en_core_web_md", disable=["ner"]) # load model without ner
        nlp.add_pipe("ner-crf-1") # add to the pipe

    if spacy_version() == 2:
        print('spacy version 2')

        # including fastext
        from fastext_loading import *

        # training using dataset
        st = time.time()
        train_examples = read_examples(train_sets[p], tokenizer=tokenizer, use_dense_features=False)
        crf_extractor.train(train_examples, dev_samples=dev_examples)
        et = time.time()
        elapsed_time = et - st
        r['time(sec)'] = elapsed_time

        test_examples = read_examples(test_sets[p], tokenizer=tokenizer, use_dense_features=False)
        r['score'] = crf_extractor.eval(test_examples)[0]
        r['stats'] = crf_extractor.eval(test_examples)[1]

        from spacy_crfsuite import CRFEntityExtractor

        # Add our CRF component to pipeline
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        pipe = CRFEntityExtractor(nlp, crf_extractor=crf_extractor)
        nlp.add_pipe(pipe)

    print('---------- Results ------------')
    '''
    text = "Apple has it's headquaters in California"
    doc = nlp(text)
    dataset1_test = []
    for ent in doc.ents:
        dataset1_test.append((ent, ent.label_))

    print(dataset1_test)'''
    print("time = ", r["time(sec)"])
    result_extractor()