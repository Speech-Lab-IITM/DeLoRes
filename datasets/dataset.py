from datasets.iemocap import  IEMOCAPTest, IEMOCAPTrain
from datasets.birdsong_dataset import BirdSongDatasetTrain, BirdSongDatasetTest
import torch

def get_dataset(downstream_task_name):
    if downstream_task_name == "birdsong_combined":
        #raise NotImplementedError
        return BirdSongDatasetTrain(), BirdSongDatasetTest()
    # elif downstream_task_name == "speech_commands_v1":
    #     return SpeechCommandsV1Train() , SpeechCommandsV1Test()
    # elif downstream_task_name == "speech_commands_v2":
    #     return SpeechCommandsV2Train() , SpeechCommandsV2Test()
    # elif downstream_task_name == "libri_100":
    #     return Libri100Train() , Libri100Test()
    # elif downstream_task_name == "musical_instruments":
    #     return MusicalInstrumentsDatasetTrain() , MusicalInstrumentsDatasetTest()
    elif downstream_task_name == "iemocap":
        return IEMOCAPTrain(),IEMOCAPTest()
    # elif downstream_task_name == "tut_urban":
    #     return TutUrbanSoundsTrain(),TutUrbanSoundsTest()
    # elif downstream_task_name == "voxceleb_v1":
    #     return Voxceleb1Train() , Voxceleb1DatasetTest()
    # elif downstream_task_name == "language_identification":
    #     return LanguageIdentificationTrain(), LanguageIdentificationTest()
    else:
        raise NotImplementedError

def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataset.no_of_classes = dataset.no_of_classes
    return train_dataset,valid_dataset
