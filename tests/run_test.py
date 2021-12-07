import unittest
import sys
import logging


if __name__ == '__main__':
    sys.path.append('../nlpatl')

    runner = unittest.TextTestRunner()

    # disable transformer's info logging
    for file_name in ['tokenization_utils', 'file_utils', 'modeling_utils', 'modeling_xlnet',
                      'configuration_utils']:
        logging.getLogger('transformers.' + file_name).setLevel(logging.ERROR)
    # disable datasets' info logging
    for file_name in ['builder']:
        logging.getLogger('datasets.' + file_name).setLevel(logging.ERROR)

    # test sub package
    test_dirs = [
        'tests/learning/',
        'tests/models/embeddings/',
        'tests/models/clustering/',
        'tests/models/classification/',
        'tests/sampling/certainty/',
        'tests/sampling/uncertainty/',
        'tests/sampling/unsupervised/',   
    ]
    for test_dir in test_dirs:
       loader = unittest.TestLoader()
       suite = loader.discover(test_dir)
       runner.run(suite)

    # test function
    suites = []
    # Embeddings
    # suites.append(unittest.TestLoader().loadTestsFromName('models.embeddings.test_sentence_transformers'))
    # suites.append(unittest.TestLoader().loadTestsFromName('models.embeddings.test_transformers'))
    # suites.append(unittest.TestLoader().loadTestsFromName('models.embeddings.test_torchvision'))

    # Clustering
    # suites.append(unittest.TestLoader().loadTestsFromName('models.clustering.test_sklearn_clustering'))

    # Classification
    # suites.append(unittest.TestLoader().loadTestsFromName('models.classification.test_sklearn_classification'))

    # Sampling
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.certainty.test_most_confidence'))
    
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.uncertainty.test_least_confidence'))
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.uncertainty.test_margin'))
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.uncertainty.test_entropy'))

    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.unsupervised.test_clustering'))

    # Labeling
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_learning'))
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_semi_supervised_learning'))
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_supervised_learning'))
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_unsupervised_learning'))

    for suite in suites:
        runner.run(suite)