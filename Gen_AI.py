# -*- coding: utf-8 -*-
"""3rd PhD D2N CycleGAN"""

import os
import time
import datetime
import imageio
from glob import glob

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

from matplotlib import pyplot as plt
from IPython import display
from termcolor import colored
from tqdm import tqdm
from IPython.display import Image
import PIL
from PIL import ImageDraw

def color_print(print_str, print_color=’green’):
    print(colored(print_str,print_color))

import math
import random

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ[“PYTHONHASHSEED”] = str(seed)
    os.environ[“TF_DETERMINISTIC_OPS”] = “1'
    print(f’setting seed to {seed}’)

class CFG:
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    resize_height = 700
    resize_width = 1200
    LAMBDA = 10
    BUFFER_SIZE = 100
    BATCH_SIZE = 2
    cache= 50
    learning_rate = 0.00025
    seed = 7

set_seed(CFG.seed)

From google.colab import drive # For colab environment
drive.mount(“/content/drive”, force_remount=False) # Mount the Google Drive.
!unzip "/content/drive/MyDrive/day_night_dataset.zip" -d "/tmp/Dataset" # Update path if needed

day_dir = ”/tmp/Dataset/day/’
night_dir = “/tmp/Dataset/night/”


def print_folder_statistics(path):
    d_image_sizes = {}
    for image_name in os.listdir(path):
        image = cv2.imread(os.path.join(path, image_name))
        d_image_sizes[image.shape] = d_image_sizes.get(image.shape, 0) + 1

    for size, count in d_image_sizes.items():
        print(f"shape: {size}\tcount: {count}")


print(f"Day images:")
print_folder_statistics(day_dir)
print("-" * 10)
print(f"Night images:")
print_folder_statistics(night_dir)
print("-" * 10)

def color_hist_visualization(image_path, figsize=(16, 4)):
    plt.figure(figsize=figsize)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.axis("off")

    colors = ["red", "green", "blue"]
    for i in range(len(colors)):
        plt.subplot(1, 4, i + 2)
        plt.hist(img[:, :, i].reshape(-1), bins=25, alpha=0.5, color=colors[i], density=True)
        plt.xlim(0, 255)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def channels_visualization(image_path, figsize=(16, 4)):
    plt.figure(figsize=figsize)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 4, 1)
    plt.imshow(np.mean(img, axis=2), cmap="gray")
    plt.axis(“off”)

    for i in range(3):
        plt.subplot(1, 4, i + 2)
        tmp_img = np.full_like(img, 0)
        tmp_img[:, :, i] = img[:, :, i]
        plt.imshow(tmp_img)
        plt.xlim(0, 255)
        plt.xticks([])
        plt.yticks([])


    plt.show()



def grayscale_visualization(image_path, figsize=(8, 4)):
    plt.figure(figsize=figsize)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis(“off”)

    plt.subplot(1, 2, 2)
    tmp_img = np.full_like(img, 0)
    for i in range(3):
        tmp_img[:, :, i] = img.mean(axis=-1)
    plt.imshow(tmp_img)
    plt.axis(“off”)

    plt.show()


def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)
    return image


def random_crop(image):
    cropped_image = tf.image.random_crop(image, size=[CFG.IMG_HEIGHT, CFG.IMG_WIDTH, 3])
    return cropped_image

def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def de_normalize(image):
    return (image * 0.5) + 0.5

def image_augmentations(image):
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if p_rotate > .8:
        image = tf.image.rot90(image, k=3)
    elif p_rotate > .6:
        image = tf.image.rot90(image, k=2) 
    elif p_rotate > .4:
        image = tf.image.rot90(image, k=1) 

    p_flip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    if p_flip > 0.7:
        image = tf.image.random_flip_left_right(image)
    elif p_flip < 0.3:
        image = tf.image.random_flip_up_down(image)

    return image

def random_jitter(image):
    image = tf.image.resize(image, size=(CFG.resize_height, CFG.resize_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    return image

def preprocess_image_train(image):
    image = load_image(image)
    image = random_jitter(image)
    image= image_augmentations(image)
    image = normalize(image)
    return image

def preprocess_image_eval(image):
    image = load_image(image)
    image = random_jitter(image)
    image = normalize(image)
    return image


def create_img_dataset(directory, image_preprocess_fn, image_extension = “jpg”, repeat=True):

    img_list = glob(directory+f’/*{image_extension}’)
    dataset = tf.data.Dataset.list_files(img_list)
    dataset = dataset.map(image_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if repeat :
        dataset = dataset.repeat()

    dataset = dataset.shuffle(CFG.BUFFER_SIZE)
    dataset = dataset.batch(CFG.BATCH_SIZE)
    return dataset


Day_Dataset = create_img_dataset(directory = day_dir,image_preprocess_fn = preprocess_image_train)
Day_eval = create_img_dataset(directory = day_dir, image_preprocess_fn = preprocess_image_eval)

fig,ax = plt.subplots(figsize=(16.8))
inp_img = next(iter(Day_Dataset))
plt.imshow(de_normalize(inp_img[0]))
plt.title(“Sample Day image”)
plt.axis(“off”)
plt.show()

Night_Dataset = create_img_dataset(directory = night_dir,image_preprocess_fn = preprocess_image_train)
Night_eval = create_img_dataset(directory = night_dir, image_preprocess_fn = preprocess_image_eval)


fig,ax = plt.subplots(figsize=(16.8))
inp_img = next(iter(Night_Dataset))
plt.imshow(de_normalize(inp_img[0]))
plt.title(“Sample Night image”)
plt.axis(“off”)
plt.show()

Train_Dataset = tf.data.Dataset.zip((Day_Dataset,Night_Dataset))

conv_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)
gamma_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


def downsample(input_layer, filters, name, size=3, strides=2, activation=tf.keras.layers.ReLU()):
    conv = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=’same’, use_bias=False,
                                  kernel_initializer=conv_initializer, name=f’encoder_{name}’)(input_layer)
    conv = tfa.layers.InstanceNormalization(axis=-1,gamma_initializer=gamma_initializer)(conv)
    conv = activation(conv)
    return conv

def upsample(input_layer, filters, name, size=3, strides=2, activation=’relu’):
    res = tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding=’same’, use_bias=False,
                                          kernel_initializer=conv_initializer, name=f’decoder_{name}’)(input_layer)
    res = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(res)
    res =  tf.keras.layers.Activation(activation)(res)
    return res

def residual_block(input_layer, size=3, strides=1, name=’block_x’):
    filters = input_layer.shape[-1]
    block = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=’same’, use_bias=False,
                                   kernel_initializer=conv_initializer, name=f’residual_{name}’)(input_layer)
    block = tf.keras.layers.Activation(“relu”)(block)
    block = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=’same’, use_bias=False,
                     kernel_initializer=conv_initializer, name=f’transformer_{name}_2')(block)

    res = tf.keras.layers.Add()([block, input_layer])
    return res

def concat_layer(layer_1,layer_2,name):
    return tf.keras.layers.Concatenate(name=name)([layer_1,layer_2])


def get_generator(num_residual_connections=6):
    input_ = tf.keras.layers.Input(shape=(CFG.IMG_WIDTH,CFG.IMG_HEIGHT,3), name=’input_layer’)

    enc1 = downsample(input_layer = input_, filters=64,  strides =  1, size=7, name=’dwn_1’)    
    enc2 = downsample(input_layer=enc1,filters= 128,size =  3, strides =  2, name=’dwn_2’)      
    enc3 = downsample(input_layer=enc2, filters=256,size =  3, strides =2, name=’dwn_3’)       
    enc4 = downsample(input_layer=enc3, filters=256,size =  3, strides =2, name=’dwn_4’)        


    x = enc4
    for n in range(num_residual_connections):
        x = residual_block(input_layer=x, name=f’res_block_{n+1}’)    


    x_skip = concat_layer(layer_1=x,layer_2=enc4,name=’skip_1’)
    dec1 = upsample(x_skip,filters=256 ,name=’upsam_1’)  

    x_skip = concat_layer(layer_1=dec1,layer_2=enc3,name=’skip_2’)
    dec_2 = upsample(x_skip, filters=128,name=’upsam_2’) 

    x_skip = concat_layer(layer_1=dec_2,layer_2=enc2,name=’skip_3’)
    dec_3 = upsample(x_skip, filters= 64,name=’upsam_3’) 

    x_skip = concat_layer(layer_1=dec_3, layer_2=enc1, name=’skip_final’)

    output = tf.keras.layers.Conv2D(filters = 3,kernel_size = 7, strides=1, padding=’same’,
                                  kernel_initializer=conv_initializer, use_bias=False, activation=’tanh’,
                                  name=’output_layer’)(x_skip)

    return tf.keras.models.Model(inputs=input_,outputs=output)


day2night_gen = get_generator()
night2day_gen = get_generator()

tf.keras.utils.plot_model(day2night_gen)

gen_output = night2day_gen(inp_img, training=False)[0]
plt.subplots(1,2,figsize=(16.8))

plt.subplot(1,2,1)
plt.imshow(gen_output.numpy().squeeze())
plt.title('Untrained Night2Day Generator output')
plt.axis(“off”)

plt.subplot(1,2,2)
plt.imshow(de_normalize(inp_img[0]))
plt.title('Original Night image')
plt.axis(“off”)

plt.show()


def PATCH_discriminator(leak_rate = 0.2):
    leaky_relu = tf.keras.layers.LeakyReLU(leak_rate)
    input_ = tf.keras.layers.Input(shape=(CFG.IMG_WIDTH,CFG.IMG_HEIGHT,3), name=’input_layer’)

    x = downsample(input_layer = input_, filters=64,  strides =  2, size=4, name=’dwn_1’,activation = leaky_relu)   
    x = downsample(input_layer = x, filters=128,  strides =  2, size=4, name=’dwn_2’,activation = leaky_relu)       
    x = downsample(input_layer = x, filters=256,  strides =  2, size=4, name=’dwn_3’,activation = leaky_relu)      
    x = downsample(input_layer = x, filters=512,  strides =  2, size=4, name=’dwn_4’,activation = leaky_relu)      
    x = downsample(input_layer = x, filters=512,  strides =  1, size=4, name=’dwn_5’,activation = leaky_relu)        

    output = tf.keras.layers.Conv2D(1, 4, strides=1, padding=’valid’, kernel_initializer=conv_initializer)(x)  

    return tf.keras.models.Model(inputs=input_,outputs=output)


day2night_disc = PATCH_discriminator()  
night2day_disc = PATCH_discriminator()  


tf.keras.utils.plot_model(day2night_disc, dpi=48)


disc_output = day2night_disc(inp_img, training=False)
plt.subplots(1,1,figsize=(8,8))

plt.imshow((disc_output.numpy().mean(axis=0)[:,:,0]),cmap=’gray’)
plt.title(“Untrained Night2Day disc output”)
plt.axis(“off”)

plt.show()


def generate_cycle(gen_1,gen_2,input_image):
    gen_img_1 = gen_1(input_image,training=True)
    gen_img_2 = gen_2(gen_img_1,training=True)
    return gen_img_1,gen_img_2


def calc_and_apply_gradients(tape, model, loss, optimizer):
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return


def discriminator_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    mae_loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * mae_loss

def identity_loss(real_image, same_image, LAMBDA):
    mae_loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * mae_loss

class CycleGAN(tf.keras.models.Model):
    def __init__(self, lambda_cycle=10):
        super(CycleGAN, self).__init__()
        self.gen_d2n = day2night_gen 
        self.gen_n2d = night2day_gen  
        self.disc_d2n = day2night_disc  
        self.disc_n2d = night2day_disc  
        self.lambda_cycle = lambda_cycle 

    def compile(self, gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn,
                 common_opt = tf.keras.optimizers.Adam(learning_rate = CFG.learning_rate,beta_1 = 0.5)):

        super(CycleGAN, self).compile()
        self.opt_gen_d2n = common_opt
        self.opt_gen_n2d = common_opt
        self.opt_disc_d2n = common_opt
        self.opt_disc_n2d = common_opt
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn


    def train_step(self, batch_data):
        day_image, night_image = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_night,cycled_day = generate_cycle(self.gen_d2n, self.gen_n2d, day_image)
            fake_day,cycled_night = generate_cycle(self.gen_n2d, self.gen_d2n, night_image)
            iden_day = self.gen_d2n(night_image, training=True)
            iden_night = self.gen_n2d(day_image, training=True)
            disc_night = self.disc_d2n(night_image, training=True)
            disc_day = self.disc_n2d(day_image, training=True)
            disc_fake_night   = self.disc_d2n(fake_night, training=True)
            disc_fake_day = self.disc_n2d(fake_day, training=True)
            night_gen_loss = self.gen_loss_fn(disc_fake_night)
            day_gen_loss = self.gen_loss_fn(disc_fake_day)
            total_cycle_loss = self.cycle_loss_fn(night_image, cycled_night, self.lambda_cycle) + self.cycle_loss_fn(day_image, cycled_day, self.lambda_cycle)
            total_gen_d2n_loss = night_gen_loss + total_cycle_loss + self.identity_loss_fn(night_image, iden_night,self.lambda_cycle)
            total_gen_n2d_loss = day_gen_loss + total_cycle_loss + self.identity_loss_fn(day_image, iden_day, self.lambda_cycle)
            night_disc_loss = self.disc_loss_fn(disc_night, disc_fake_night) 
            day_disc_loss = self.disc_loss_fn(disc_day, disc_fake_day)       

        _ = calc_and_apply_gradients(tape=tape, model= self.gen_d2n, loss = total_gen_d2n_loss, optimizer = self.opt_gen_d2n)
        _ = calc_and_apply_gradients(tape=tape, model= self.gen_n2d, loss = total_gen_n2d_loss, optimizer = self.opt_gen_n2d)
        _ = calc_and_apply_gradients(tape=tape, model= self.disc_d2n, loss = night_disc_loss, optimizer = self.opt_disc_d2n)
        _ = calc_and_apply_gradients(tape=tape, model= self.disc_n2d, loss = day_disc_loss, optimizer = self.opt_disc_n2d)

        return {“gen_D2N_loss”: total_gen_d2n_loss, “gen_N2D_loss”: total_gen_n2d_loss,
                “disc_day_loss”: day_disc_loss, “disc_night_loss”: night_disc_loss}

gan = CycleGAN()
gan.compile(gen_loss_fn=generator_loss, disc_loss_fn=discriminator_loss,
            cycle_loss_fn=calc_cycle_loss, identity_loss_fn=identity_loss)

def scheduler(epoch, lr, decay_rate = 0.05, warm_up_period = 10):
    if epoch < warm_up_period:
        return lr
    elif (epoch > warm_up_period and epoch<40):
        return lr * tf.math.exp(decay_rate)
    else:
        return lr * tf.math.exp(decay_rate*2)



lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 0)

early_stop = tf.keras.callbacks.EarlyStopping(monitor = “gen_N2D_loss”, mode = “min”, patience = 10, restore_best_weights = True)


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, patience=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.n2d_loss = np.Inf
        self.d2n_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        n2d_loss=logs.get(“gen_N2D_loss”)
        d2n_loss=logs.get(“gen_D2N_loss”)

        if (np.less(n2d_loss, self.n2d_loss) and np.less(d2n_loss, self.d2n_loss)):
            self.d2n_loss = d2n_loss
            self.n2d_loss = n2d_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))



def display_samples(ds, n_samples):
    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        example_sample = next(ds_iter)
        plt.subplot(121)
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis(“off”)
        plt.show()


def display_generated_samples(ds, model, n_samples):
    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        example_sample = next(ds_iter)
        generated_sample = model.predict(example_sample)

        f = plt.figure(figsize=(16.8))

        plt.subplot(121)
        plt.title(“Input image”)
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis(“off”)

        plt.subplot(122)
        plt.title('Generated image')
        plt.imshow(generated_sample[0] * 0.5 + 0.5)
        plt.axis(“off”)
        plt.show()


def evaluate_cycle(ds, generator_a, generator_b, n_samples=1):
    fig, axes = plt.subplots(n_samples, 3, figsize=(22, (n_samples*6)))
    axes = axes.flatten()

    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        idx = n_sample*3
        example_sample = next(ds_iter)
        generated_a_sample = generator_a.predict(example_sample)
        generated_b_sample = generator_b.predict(generated_a_sample)

        axes[idx].set_title(“Input image”, fontsize=18)
        axes[idx].imshow(example_sample[0] * 0.5 + 0.5)
        axes[idx].axis(“off”)

        axes[idx+1].set_title(“Generated image”, fontsize=18)
        axes[idx+1].imshow(generated_a_sample[0] * 0.5 + 0.5)
        axes[idx+1].axis(“off”)

        axes[idx+2].set_title(“Cycled image”, fontsize=18)
        axes[idx+2].imshow(generated_b_sample[0] * 0.5 + 0.5)
        axes[idx+2].axis(“off”)

    plt.show()


def predict_and_save(input_ds, generator_model, output_path):
    i = 1
    for img in input_ds:
        prediction = generator_model(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8) 
        im = PIL.Image.fromarray(prediction)
        im.save(f’{output_path}{str(i)}.jpg’)
        i += 1


class GANMonitor(tf.keras.callbacks.Callback):

    def __init__(self, num_img=1, day_paths=’generated_day’, night_paths=’generated_night’):
        self.num_img = num_img
        self.day_paths = day_paths
        self.night_paths = night_paths

        if not os.path.exists(self.day_paths):
            os.makedirs(self.day_paths)

        if not os.path.exists(self.night_paths):
            os.makedirs(self.night_paths)


    def on_epoch_end(self, epoch, logs=None):
        for i, img in enumerate(Day_eval.take(self.num_img)):
            prediction = day2night_gen(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f’{self.night_paths}/generated_{i}_{epoch+1}.png’)


        for i, img in enumerate(Night_eval.take(self.num_img)):
            prediction = night2day_gen(img, training=False)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            prediction = PIL.Image.fromarray(prediction)
            prediction.save(f’{self.day_paths}/generated_{i}_{epoch+1}.png’)


EPOCHS = 50
callbacks = [lr_scheduler,GANMonitor(),CustomEarlyStopping(patience = 10)]
steps_per_epoch = 200

history = gan.fit(Train_Dataset, epochs = EPOCHS, steps_per_epoch=steps_per_epoch, callbacks = callbacks)

loss_results_df = pd.DataFrame(history.history).applymap(np.mean)

plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.plot(loss_results_df.index, loss_results_df[“gen_D2N_loss”], color=’g’, label=’Loss D2N Generator’)
plt.plot(loss_results_df.index, loss_results_df[“gen_N2D_loss”], color=’r’, label=’Loss N2D Generator’)
plt.plot(loss_results_df.index, loss_results_df[“disc_day_loss”], color=’b’, label=’Loss D2N Discriminator’)
plt.plot(loss_results_df.index, loss_results_df[“disc_night_loss”], color=’m’, label=’Loss N2D Discriminator’)
plt.title("Training Losses")
plt.legend(loc=’best’)
plt.show()


def create_gif(images_path, gif_path):
    images = []
    filenames = glob(images_path)
    filenames.sort(key=lambda x: int(“.join(filter(str.isdigit, x))))
    for epoch, filename in enumerate(filenames):
        img = PIL.Image.open(filename)
        ImageDraw.Draw(img).text((0, 0),f'Epoch {epoch+1}')
        images.append(img)
    imageio.mimsave(gif_path, images, fps=2) 



create_gif(“./generated_day/*.png”, “day.gif”)
print(“Training progress for Night -> Day Generator”)
display.Image(“./day.gif”)

create_gif(“./generated_night/*.png”, “night.gif”)
print(“Training progress for Day-> Generator”)
display.Image(“./night.gif”)


display_generated_samples(Day_eval.take(2), day2night_gen, 2)
display_generated_samples(Night_eval.take(2), night2day_gen, 2)
display_generated_samples(Night_eval.take(2), day2night_gen, 2)
display_generated_samples(Day_eval.take(2), night2day_gen, 2)


evaluate_cycle(Day_eval.take(2), day2night_gen, night2day_gen, n_samples=2)
evaluate_cycle(Night_eval.take(2), night2day_gen, day2night_gen,n_samples=2)


import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class PhotoDataset(Dataset):
    def __init__(self, photo_dir, size=(256, 256), normalize=True):
        super().__init__()
        self.photo_dir = photo_dir
        self.photo_idx = dict()
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()
            ])
        for i, fl in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = fl

    def __getitem__(self, idx):
        photo_path = os.path.join(self.photo_dir, self.photo_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        return photo_img

    def __len__(self):
        return len(self.photo_idx.keys())

