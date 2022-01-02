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
        'tests/sampling/',
        'tests/sampling/certainty/',
        'tests/sampling/uncertainty/',
        'tests/sampling/clustering/',
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
    # suites.append(unittest.TestLoader().loadTestsFromName('models.clustering.test_sklearn_extra_clustering'))

    # Classification
    # suites.append(unittest.TestLoader().loadTestsFromName('models.classification.test_sklearn_classification'))
    # suites.append(unittest.TestLoader().loadTestsFromName('models.classification.test_xgboost_classification'))

    # Sampling
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.test_sampling'))

    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.certainty.test_most_confidence'))
    
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.uncertainty.test_entropy'))
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.uncertainty.test_least_confidence'))
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.uncertainty.test_margin'))
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.uncertainty.test_mismatch'))
    
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.clustering.test_nearest_mean'))
    # suites.append(unittest.TestLoader().loadTestsFromName('sampling.clustering.test_farthest'))

    # Labeling
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_learning'))
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_mismatch_farthest_learning'))
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_semi_supervised_learning'))
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_supervised_learning'))
    # suites.append(unittest.TestLoader().loadTestsFromName('learning.test_unsupervised_learning'))

    for suite in suites:
        runner.run(suite)