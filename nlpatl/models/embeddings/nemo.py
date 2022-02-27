from typing import List, Dict
import numpy as np

try:
    import librosa
except ImportError:
    # No installation required if not using this function
    pass
try:
    import torch
except ImportError:
    # No installation required if not using this function
    pass
try:
    import nemo
    from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
except ImportError:
    # No installation required if not using this function
    pass

from nlpatl.models.embeddings.embeddings import Embeddings


class Nemo(Embeddings):
    """
    A wrapper of nemo class.

    :param model_name_or_path: nemo model name. Verifeid. `titanet_large`, 
        `speakerverification_speakernet` and `ecapa_tdnn`. Refer to 
        https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/speaker_recognition/intro.html
    :type model_name_or_path: str
    :param batch_size: Batch size of data processing. Default is 16
    :type batch_size: int
    :param target_sr: Sample rate. Audio will be resample to this value.
    :type target_sr: int
    :param device: Device for processing data
    :type device: str
    :param name: Name of this embeddings
    :type name: str

    >>> import nlpatl.models.embeddings as nme
    >>> model = nme.Nemo()
    """

    def __init__(
        self,
        model_name_or_path: str = 'titanet_large',
        batch_size: int = 16,
        target_sr: int = 16000,
        device: str = 'cuda',
        name: str = "nemo",
    ):
        super().__init__(batch_size=batch_size, name=name)

        self.model_name_or_path = model_name_or_path
        self.target_sr = target_sr
        self.device = device

        self.model = self.get_model(
            model_name=model_name_or_path
        ).to(self.device)
        self.model.freeze()

    def get_available_model_names(self, model_types):
        available_model_info = {}
        for model_type in model_types:
            if model_type == 'sr':
                for sr_model_info in EncDecSpeakerLabelModel.list_available_models():
                    available_model_info[sr_model_info.pretrained_model_name] = 'sr'
            else:
                raise ValueError(
                    'Does not support this model_type ({}) yet.'.format(model_type)
                )

        return available_model_info

    def get_model(self, model_name):
        # TODO: Support speaker recognition only now.
        available_model_info = self.get_available_model_names(
            model_types=['sr']
        )

        if model_name not in available_model_info.keys():
            raise ValueError(
                'Does not support this model ({}) yet. Supporting {} now'.format(
                    model_name, available_model_info.keys()
                )
            )

        model_type = available_model_info[model_name]

        if model_type == 'sr':
            # TODO: Support custom trained file
            return EncDecSpeakerLabelModel.from_pretrained(model_name)
        else:
            raise ValueError(
                'Does not support this model_type ({}) or model_name ({}) yet.'.format(
                    model_type, model_name
                )
            )

    def _resample(self, data, target_sr):
        max_len = -1
        audios = []

        # TODO: Performance tuning. Skip looping if no resample is needed
        for i in range(len(data)):
            audio = data[i][0]
            sr = data[i][1]

            if sr != target_sr:
                audio = librosa.core.resample(
                    audio, sr, target_sr
                )

            audios.append(audio)
            max_len = max(max_len, audio.shape[0])

        return audios, max_len

    def _pad(self, audios, max_len):
        for i in range(len(audios)):
            audio = audios[i]

            diff = max_len - audio.shape[0]
            if diff > 0:
                audios[i] = np.pad(
                    audio, (0, diff), 'constant', constant_values=0
                )

        return audios

    def _to_tensor(self, audios, device):
        audio_signal, audio_signal_len = (
            torch.tensor(np.array(audios), device=device),
            torch.tensor([audios[0].shape[0]] * len(audios), device=device),
        )

        return audio_signal, audio_signal_len

    def convert(self, x: List[Dict[np.ndarray, int]]) -> np.ndarray:
        results = []
        for batch_inputs in self.batch(x, self.batch_size):
            audios, max_len = self._resample(batch_inputs, self.target_sr)
            audios = self._pad(audios, max_len)

            audio_signal, audio_signal_len = self._to_tensor(audios, self.device)

            del audios

            with torch.no_grad():
                _, embs = self.model.forward(
                    input_signal=audio_signal, 
                    input_signal_length=audio_signal_len
                )

            del audio_signal, audio_signal_len, _

            if self.device != 'cpu':
                embs = embs.cpu()

            results.extend(embs.numpy())

        return np.array(results)
