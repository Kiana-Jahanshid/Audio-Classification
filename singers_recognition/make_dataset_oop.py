import pydub
import matplotlib.pyplot as plt
import os

class Make_Dataset():
    def __init__(self):
        pass

    def merge_vioces(self):
        file_names = ["chaartar","chavoshi","sirvan","xaniar","yegane"]
        for i in range(len(file_names)):
            voice_1 = pydub.AudioSegment.from_file(f"singers_vocal/{file_names[i]}_1.wav" , format="wav")
            voice_2 = pydub.AudioSegment.from_file(f"singers_vocal/{file_names[i]}_2.wav" , format="wav")
            result  = voice_1 + voice_2
            f_result = result.export(f"data/{file_names[i]}.wav")

    def omit_silence(self , data_path , wav_data_path):
        for file in os.listdir(data_path):
            audio = pydub.AudioSegment.from_file(os.path.join(data_path , file))            
            silent_chunks = pydub.silence.split_on_silence(audio , min_silence_len=2000 , silence_thresh=-45) 
            result = sum(silent_chunks)  
            result.export(wav_data_path + file)

    def divide_into_1sec(self , wav_data_path , dataset_path):
        for file in os.listdir(wav_data_path):
            audio = pydub.AudioSegment.from_file(os.path.join(wav_data_path , file))
            chunks = pydub.utils.make_chunks(audio , 1000)
            # create a folder for each person
            person_name = file.split(".")[0]
            os.makedirs(os.path.join(dataset_path , person_name) , exist_ok=True)
          
            for i , chunk in enumerate(chunks) :
                if len(chunk) >= 1000 : 
                    chunk.export(os.path.join(dataset_path , person_name , f"voice_{i}.wav") , format="wav")


if __name__ == "__main__" :
    class_obj= Make_Dataset()
    class_obj.merge_vioces()
    class_obj.omit_silence(data_path="data" , wav_data_path="wav_data/")
    class_obj.divide_into_1sec(wav_data_path="wav_data" , dataset_path="dataset")