import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.layers import Dense, Lambda, Convolution2D, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.initializers import glorot_uniform
from numpy import save
from tensorflow.keras.preprocessing import image
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects


n_train_imgs = 10000
n_val_imgs = 5000
training_steps = n_train_imgs/10
validation_steps = n_val_imgs/10
img_size = 128
epochs = 0
latent_dim = 80
load_weights = True
alpha = 1         #Faktor Rekonstruktionsverlust
beta = K.variable(0.0)            #Faktor Latentetr Regularisierungsverlust
beta_max = 1
wu_epochs = epochs//1

model = 'Residual_VAE_4'
path = '/home/itoengeljon/Masterarbeit/Software/'
img_path = '/home/itoengeljon/Masterarbeit/Bilder5'
directory = path + model

if not os.path.exists(directory):
    os.makedirs(directory)

checkpoint_filepath = '/home/itoengeljon/Masterarbeit/Software/' + model + '/' + model + '_2_weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_filepath)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
training_dir = img_path + '/Test'
validation_dir = img_path + '/Test'
test_dir = img_path + '/Test'

batch_size = 10


train_generator = image_generator.flow_from_directory(
    directory=training_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=10,
    class_mode="input",
    shuffle=True,
    seed=42
)

validation_generator = image_generator.flow_from_directory(
    directory=validation_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=10,
    class_mode="input",
    shuffle=True,
    seed=42
)

test_generator = image_generator.flow_from_directory(
    directory=test_dir,
    target_size=(img_size, img_size),
    color_mode="grayscale",
    batch_size=1,
    class_mode="input",
    shuffle=False,
    seed=42
)

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon               #Reparameterrization Trick

    def get_config(self):
        config = super().get_config()
        config['z_mean'] = z_mean
        config['z_log_var'] = z_log_var
        return config


class ResnetEncoder(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetEncoder, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, kernel_size=(1, 1), padding='same')
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class ResnetDecoder(tf.keras.Model):
  def __init__(self, kernel_size, filters):
    super(ResnetDecoder, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2DTranspose(filters1, (1, 1))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2DTranspose(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2DTranspose(filters3, kernel_size=(1, 1), padding='same')
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)





input_img = tf.keras.Input(shape=(img_size, img_size, 1), name='input_img')
encoder_layers = Sequential(
    [
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        ResnetEncoder(9, (64, 64, 64)),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        Flatten(),
        Dense(latent_dim)
    ], name='encoding_layers'
)

encoded = encoder_layers(input_img)

z_mean = Dense(latent_dim, name="z_mean")(encoded)
z_log_var = Dense(latent_dim, name="z_log_var")(encoded)        #z_mean und z_log_var sind zwei unterschiedliche Layer -> unterschiedliche Gewichte
z = Sampling()([z_mean, z_log_var])
encoder = Model(input_img, [z_mean, z_log_var, z])
encoder.summary()

decoder_layers = Sequential(
    [
        Dense(8 * 8 * 64, activation='relu'),
        Reshape((8, 8, 64)),
        Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        ResnetDecoder(9, [64, 64, 64]),
        Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='sigmoid'),
    ], name='decoder_layers'
)

decoded = decoder_layers(z)
decoder_layers.summary()
#decoder = Model(z, decoded)
def loss_function():
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(input_img, decoded))
    kl_loss = -0.5*(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(kl_loss)
    return alpha * reconstruction_loss + beta * kl_loss

def warmup(epoch):
    if wu_epochs != 0:
        value = (1/wu_epochs) * epoch * beta_max * (epoch <= wu_epochs) + 1 * beta_max * (epoch > wu_epochs)
    else:
        value = beta_max
    print('beta: ', value)
    K.set_value(beta, value)

warmup_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs : warmup(epoch))

checkpoint = tf.keras.callbacks.ModelCheckpoint(                #Abspeichern von Gewichten, ohne Modell
    filepath=checkpoint_filepath,
    monitor='loss',
    verbose=0,
    save_weights_only=True,
    save_best_only=True,
    mode='min',
    save_freq='epoch'
)

earlystop = tf.keras.callbacks.EarlyStopping(                   #Vorzeitiges Beenden von Training falls kein Fortschritt zum Verhindern von Overfitting
    monitor='val_loss',
    min_delta=0.000,
    patience=20,
    verbose=1,
    restore_best_weights=True
)

logdir = '/home/itoengeljon/Masterarbeit/Software/'+model+'/log'                      #Verzeichnis für Aufzeichnung Trainingsfortschritt

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    write_graph=True,
    update_freq='epoch'
    )

vae = tf.keras.Model(input_img, decoded)
vae.summary()
#vae_loss = K.mean(loss_function())
vae.add_loss(K.mean(loss_function()))
vae.compile(optimizer=tf.keras.optimizers.Adam())
if load_weights is True:
    vae.load_weights(checkpoint_filepath)
else:
    pass

vae.fit(train_generator,
                epochs=epochs,
                steps_per_epoch=training_steps,
                shuffle=True,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=[earlystop, checkpoint, tensorboard_callback, warmup_callback]
)



class Predictor():
    def __init__(self, feature, n_imgs=500):
        self.feature = feature
        self.n_imgs = n_imgs

    latent_var = []
    def predictor(self):                                                    #Bilder einzeln aus Test Ordner
        '''for n in range(self.n_imgs):
            image = Image.open(img_path + '/Test/test/test'+str(n)+'.jpeg')
            image = np.asarray(image)
            image = tf.expand_dims(image, axis=0)
            image = tf.reshape(image, (1, img_size, img_size, 1))                        # Reshape: Stack, x, y, Farbe
            z_mean, _, z = encoder.predict(image)
            z_mean = np.asarray(z_mean)                                     #predict latenter Vektor
            z = np.asarray(z)
            z_mean = z_mean.flatten()                                       #latenten Vektor umformen zu 1xn, damit von t-sne und pca verarbeitet werden kann
            z = z.flatten()
            self.latent_var.append(z_mean)'''

        for i in range(len(os.listdir(img_path + '/Test/test'))):
            z_mean,_,z = encoder.predict(test_generator[i], batch_size=1)
            latent_var = np.array(z_mean)
            self.latent_var.append(latent_var)

    def get_labels(self):
        label_list = pd.read_csv(img_path + '/Test/labels.csv')
        if self.feature == 'n_patches':
            labels = np.array(label_list.n_patches[:self.n_imgs].astype('int32'))
            return labels
        if self.feature == 'shapes':
            labels = []
            shapes_list = label_list.filter(like=' patch', axis=1)              #filtert alle Spalten mit ' patch' im Kopf
            for n in range(len(shapes_list.iloc[:, 1])):
                if {1, 2, 3}.issubset(shapes_list.loc[n]):                      #überprüfen ob alle drei Formen in Bild vorhanden sind
                    labels.append('rect-circ-tri')
                elif {2, 3}.issubset(shapes_list.loc[n]):                       #falls Bild nicht alle drei Formen enthält, Überprüfung auf zwei Formen
                    labels.append('circ-tri')
                elif {1, 3}.issubset(shapes_list.loc[n]):
                    labels.append('rect-tri')
                elif {1, 2}.issubset(shapes_list.loc[n]):
                    labels.append('rect-circ')
                elif any(shapes_list.loc[n] == 1):
                    labels.append('rect')
                elif any(shapes_list.loc[n] == 2):
                    labels.append('circ')
                elif any(shapes_list.loc[n] == 3):
                    labels.append('tri')
                else:
                    labels.append(0)
            labels = np.array(labels)
            return labels


    def pca(self, n_components):
        pca = PCA(n_components)
        pca_result = pca.fit_transform(self.latent_var)
        pca_df = pd.DataFrame(data=None)
        for n in range(n_components):
            pca_df.insert(loc=n, column='pca'+str(n+1), value=None)
            pca_df['pca'+str(n+1)] = pca_result[:, n]
        print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
        return np.array(pca_df)

    def tsne(self):
        latent_var = self.latent_var
        latent_var = np.array(latent_var)
        latent_var = latent_var.reshape((500, latent_dim))
        tsne = TSNE(n_components=2, random_state=0)
        tsne_df = tsne.fit_transform(latent_var)
        return tsne_df

    def feature_scatter(self, method):                                             #Plot von Daten gegen Feature
        # choose a color palette with seaborn.
        labels = self.get_labels()
        label_names = np.unique(labels)
        num_classes = len(np.unique(labels))
        palette = np.array(sns.color_palette("hls", num_classes))
        if method == 'pca':
            data = self.pca(n_components=2)
        if method == 'tsne':
            data = self.tsne()
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aqua', 'orange', 'purple', 'magenta', 'brown', 'grey', 'yellowgreen', 'midnightblue'
        for i, c, label in zip(range(num_classes), colors, label_names):
            plt.scatter(data[labels == label_names[i], 0], data[labels == label_names[i], 1], c=c, label=label)
        #sc = ax.scatter(x=pca_data.iloc[:, 0], y=pca_data.iloc[:, 1], lw=0, s=40, c=palette[labels.astype(np.int)])
        ax.axis('off')
        ax.axis('tight')
        plt.legend()
        plt.show()
        return f, ax, #sc, #txts

    def img_algebra(self, img_num):
        img_num_1, img_num_2, img_num_3 = img_num
        latent_var = self.latent_var
        pred_sum = decoder_layers.predict(latent_var[img_num_1] - latent_var[img_num_2] + latent_var[img_num_3])
        predictions = []
        for i in range(len(latent_var)):
            pred = decoder_layers.predict(latent_var[i])
            predictions.append(pred)
        img_1 = np.reshape(predictions[img_num_1], (img_size, img_size, 1))
        img_2 = np.reshape(predictions[img_num_2], (img_size, img_size, 1))
        img_3 = np.reshape(predictions[img_num_3], (img_size, img_size, 1))
        img_4 = np.reshape(pred_sum, (img_size, img_size, 1))
        plot = plt.figure()
        plot.add_subplot(1, 4, 1)
        plt.imshow(img_1, cmap='gray')
        plot.add_subplot(1, 4, 2)
        plt.imshow(img_2, cmap='gray')
        plot.add_subplot(1, 4, 3)
        plt.imshow(img_3, cmap='gray')
        plot.add_subplot(1, 4, 4)
        plt.imshow(img_4, cmap='gray')
        plt.show()


P = Predictor(feature='shapes', n_imgs=500)
P.predictor()
P.img_algebra([28, 27, 31])
P.feature_scatter(method='tsne')

