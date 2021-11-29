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
        'tests/models/embeddings/',
        'tests/models/clustering/',
        'tests/models/classification/',
        'tests/models/learning/',
        'tests/models/learning/unsupervised/',
        'tests/models/learning/uncertainty/',
    ]
    for test_dir in test_dirs:
       loader = unittest.TestLoader()
       suite = loader.discover(test_dir)
       runner.run(suite)

    # test function
    suites = []
    # Embeddings
    # suites.append(unittest.TestLoader().loadTestsFromName('models.embeddings.test_transformers'))

    # Clustering
    # suites.append(unittest.TestLoader().loadTestsFromName('models.clustering.test_sklearn_clustering'))

    # Classification
    # suites.append(unittest.TestLoader().loadTestsFromName('models.classification.test_sklearn_classification'))

    # Labeling
    # suites.append(unittest.TestLoader().loadTestsFromName('models.learning.test_learning'))

    # suites.append(unittest.TestLoader().loadTestsFromName('models.learning.unsupervised.test_clustering'))
    
    # suites.append(unittest.TestLoader().loadTestsFromName('models.learning.uncertainty.test_uncertainity'))
    # suites.append(unittest.TestLoader().loadTestsFromName('models.learning.uncertainty.test_margin'))
    # suites.append(unittest.TestLoader().loadTestsFromName('models.learning.uncertainty.test_entropy'))

    for suite in suites:
        runner.run(suite)