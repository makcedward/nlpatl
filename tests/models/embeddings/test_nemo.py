import unittest
import glob
import os
import librosa

from nlpatl.models.embeddings.nemo import Nemo


class TestModelEmbeddingsNemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        res_file_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', 'res', 'audio', ''
        )
        file_paths = glob.glob(res_file_dir + '*')

        cls.data = [librosa.load(f) for f in file_paths]

    # def test_convert(self):
        for model in ['titanet_large', 'speakerverification_speakernet', 'ecapa_tdnn']:
            embs_model = Nemo(model_name_or_path=model, device='cpu')
            embs = embs_model.convert(self.data)

            assert len(self.data) == len(embs), \
                "Number of input does not equal to number of outputs"

    def test_unsupport_model(self):
        with self.assertRaises(Exception) as error:
            embs_model = Nemo(model_name_or_path='unsupport', device='cpu')

        assert "Does not support this " in str(
            error.exception
        ), "Unable to handle unsupported embeddings model"
