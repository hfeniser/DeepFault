from keras.models import model_from_json

def load_model():
    json_file = open('simple_mnist_fnn.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into model
    model.load_weights("simple_mnist_fnn.h5")

    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model
