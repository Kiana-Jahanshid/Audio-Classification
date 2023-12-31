import os
import tensorflow as tf
import numpy as np
import pydub


class Train :
    def __init__(self):
        pass

    def load_data(self):
        dataset_path = "./dataset/"

        train_data = tf.keras.utils.audio_dataset_from_directory(
        dataset_path,
        batch_size=8,
        shuffle=True,
        validation_split=0.2,
        subset="training",

        output_sequence_length= 48000,
        ragged= False,
        labels="inferred",
        label_mode="categorical",
        sampling_rate= None,
        seed=59
        )


        validation_data = tf.keras.utils.audio_dataset_from_directory(
        dataset_path,
        batch_size=8,
        shuffle=True,
        validation_split=0.2,
        subset="validation",

        # new parameter for AUDIO task
        output_sequence_length= 48000,
        ragged= False,
        labels="inferred",
        label_mode="categorical",
        sampling_rate= None,
        seed=59
        )

        return train_data , validation_data
    

    def create_model(self):
        model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size = 80,strides = 16, activation = "relu", input_shape = (48000, 1)),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(32, kernel_size = 3, activation = "relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(64, kernel_size = 3,activation = "relu"),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(64, kernel_size = 3,activation = "relu"),
        tf.keras.layers.MaxPooling1D(4),
        tf.keras.layers.Conv1D(64, kernel_size = 3,activation = "relu"),
        tf.keras.layers.MaxPooling1D(4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5 , activation="softmax")
        ])

        return model
    

    def compile_model(self , model):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss = tf.keras.losses.categorical_crossentropy ,
              metrics="accuracy"
              )
        

    def save_best_model(self):
        filepath = 'model/t.h5'
        model_callback = tf.keras.callbacks.ModelCheckpoint( filepath=filepath ,monitor='val_accuracy',mode='max',save_best_only=True)
        return model_callback
    
    def train(self ,model , train_data , validation_data , model_callback):

        output = model.fit( train_data , validation_data=validation_data , epochs=20 ,callbacks=[model_callback])


    def predict(self , model):
        pred_list = []
        for i , file in enumerate(os.listdir("dataset2/vocals")):         
            file_path = f"dataset2/vocals/voice_{i}.wav"
            x = tf.io.read_file(file_path)
            x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=48000,)
            x = tf.squeeze(x , axis=-1)
            x = x[tf.newaxis, ...]
            classes_labels = np.array(['chaartar', 'chavoshi', 'sirvan', 'xaniar', 'yegane'])
            predictions = model.predict(x)
            predicted_class_index = np.argmax(predictions)
            predicted_class = classes_labels[predicted_class_index]
            pred_list.append(predicted_class)

        predicted_class = max(pred_list ,key=pred_list.count)
        return print(f'The owner of this song is : {predicted_class} ')


if __name__ == "__main__" :
    classtrain = Train()
    train_data , validation_data = classtrain.load_data()
    model = classtrain.create_model()
    classtrain.compile_model(model)
    model_callback = classtrain.save_best_model()
    classtrain.train(model , train_data , validation_data , model_callback)
    classtrain.predict(model)
