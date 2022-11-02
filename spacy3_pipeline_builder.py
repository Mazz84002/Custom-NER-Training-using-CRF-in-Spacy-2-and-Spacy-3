import spacy

# load the model
def build_spacy3_pipeline():
    # create a blank spacy model
    pipes = ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
    nlp = spacy.load('/Users/mazz/Documents/Programming/Python/Mini-Project/Script/Fastext_models3')
    # create a reference model
    nlp_ref = spacy.load('en_core_web_md')
    for pipe in pipes:
        if pipe in ['ner']:
            nlp.add_pipe(pipe, source=nlp_ref)
        else:
            nlp.add_pipe(pipe)
    return nlp
