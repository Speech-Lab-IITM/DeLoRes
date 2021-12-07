from datasets_byol_gen.musical_instruments_avg import MusicalInstrumentsTrain, MusicalInstrumentsTest
from datasets_byol_gen.language_identification_avg import LanguageIdentificationTrain, LanguageIdentificationTest
from datasets_byol_gen.voxceleb_avg import Voxceleb1DatasetTrain, Voxceleb1DatasetTest
import torch

def get_dataset(downstream_task_name,aug = None):
    
          
    if downstream_task_name == "musical_instruments":
        return MusicalInstrumentsTrain(tfms=aug) , MusicalInstrumentsTest(tfms=aug)
    elif downstream_task_name == "voxceleb_v1":
        return Voxceleb1DatasetTrain(tfms=aug) , Voxceleb1DatasetTest(tfms=aug)   
    elif downstream_task_name == "language_identification":
        return LanguageIdentificationTrain(tfms=aug), LanguageIdentificationTest(tfms=aug)                 
    else:
        raise NotImplementedError

