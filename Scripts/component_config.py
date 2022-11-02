# contains features used to train crf in all the datasets

def return_custom_components():
    component_config = {
        "features": [
            [
                "low",
                "title",
                "upper",
                "pos",
                "pos2"
            ],
            [
                "low",
                "bias",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pos",
                "pos2"
            ],
            [
                "low",
                "title",
                "upper",
                "pos",
                "pos2"
            ],
        ],
        "c1": 0.01,
        "c2": 0.22
    }
    return component_config