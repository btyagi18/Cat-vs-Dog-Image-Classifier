from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

def train_model(model, X_train, y_train, X_val, y_val):
    datagen = ImageDataGenerator(
        rotation_range=30,
        zoom_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=10
    )

    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/model.h5")

    return history
