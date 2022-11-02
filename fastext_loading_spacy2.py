import spacy
import subprocess
import sys

model_name = "fastext"
word_vectors = "/Users/mazz/Documents/Programming/Python/Mini-Project/Script/Fastext_models/cc.en.300.vec"

def spacy_version():
    ver = int(str(spacy.__version__)[0])
    return ver

def load_word_vectors(model_name, word_vectors):
    if spacy_version() == 3:
        pass

    if spacy_version() == 2:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "spacy",
                "init-model",
                "en",
                model_name,
                "--vectors-loc",
                word_vectors
            ]
        )


load_word_vectors(model_name=model_name, word_vectors=word_vectors)