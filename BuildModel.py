#Import libraries

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

# include_top = False - to remove the last few layer of the model for hypetuning 
vgg = VGG16(include_top=False)

#import function from def_1
import def_1 as ld


# Data processing using tensorflow before loading into the model

train_images = tf.data.Dataset.list_files('imagewebcam\Train\images\*.jpg', shuffle=False)
train_images = train_images.map(ld.load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('imagewebcam\Test\images\*.jpg', shuffle=False)
test_images = test_images.map(ld.load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('imagewebcam\Val\images\*.jpg', shuffle=False)
val_images = val_images.map(ld.load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)


train_labels = tf.data.Dataset.list_files('imagewebcam\Train\labels\*.json', shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(ld.load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('imagewebcam\Test\labels\*.json', shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(ld.load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('imagewebcam\Val\labels\*.json', shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(ld.load_labels, [x], [tf.uint8, tf.float16]))


train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)


test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(1300)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1000)
val = val.batch(8)
val = val.prefetch(4)


# Building the model with hypetuning 

def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model  
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    facedetect = Model(inputs=input_layer, outputs=[class2, regress2])
    return facedetect

facedetect = build_model()

# Tuning the optimizer

batches_per_epoch = len(train)
lr_decay = (1./0.75 -1)/batches_per_epoch

opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = ld.localization_loss

# define model training details

class FaceDetect(Model): 
    def __init__(self, facetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = facetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

#Training the Model

model = FaceDetect(facedetect)

model.compile(opt, classloss, regressloss)

logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=40, validation_data=val, callbacks=[tensorboard_callback])

# Save the Model

facedetect.save('facedetect.h5')