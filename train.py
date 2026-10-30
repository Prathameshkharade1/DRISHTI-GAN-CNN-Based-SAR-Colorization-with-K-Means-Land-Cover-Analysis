import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random

tf.random.set_seed(42)

# --- 1. DEFINE CONSTANTS AND PATHS ---
IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 400
BATCH_SIZE = 1
OUTPUT_CHANNELS = 3
LAMBDA = 100

# --- 2. DATA LOADING AND PREPROCESSING ---
def load_image_paths(base_path):
    input_img_paths = []
    real_img_paths = []
    categories = ['urban', 'grassland', 'barrenland', 'agri']
    
    print("Searching for image pairs using filename convention...")
    for category in categories:
        s1_dir = os.path.join(base_path, category, 's1')
        s2_dir = os.path.join(base_path, category, 's2')
        
        if not os.path.exists(s1_dir):
            print(f"Warning: Directory '{s1_dir}' not found. Skipping.")
            continue

        for s1_filename in os.listdir(s1_dir):
            if not s1_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            s2_filename = s1_filename.replace('_s1_', '_s2_')
            s1_filepath = os.path.join(s1_dir, s1_filename)
            s2_filepath = os.path.join(s2_dir, s2_filename)
            
            if os.path.exists(s2_filepath):
                input_img_paths.append(s1_filepath)
                real_img_paths.append(s2_filepath)
    return input_img_paths, real_img_paths

def load_and_preprocess_image(input_path, real_path):
    input_image = tf.io.read_file(input_path)
    input_image = tf.io.decode_image(input_image, channels=3, expand_animations=False)
    real_image = tf.io.read_file(real_path)
    real_image = tf.io.decode_image(real_image, channels=3, expand_animations=False)
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    input_image = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH])
    real_image = tf.image.resize(real_image, [IMG_HEIGHT, IMG_WIDTH])
    input_image = (input_image / 127.5) - 1.0
    real_image = (real_image / 127.5) - 1.0
    return input_image, real_image

# --- 3. MODEL ARCHITECTURE (GENERATOR & DISCRIMINATOR) ---
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                           kernel_initializer=initializer, activation='tanh')
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# --- 4. LOSS FUNCTIONS AND OPTIMIZERS ---
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

# --- 5. TRAINING STEP AND VISUALIZATION ---
def _set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _save_raw_prediction(prediction, preview_dir, epoch_idx):
    """Save the raw generated image as PNG (not the matplotlib figure)."""
    pred_01 = (prediction[0] * 0.5 + 0.5)
    pred_uint8 = tf.cast(tf.round(pred_01 * 255.0), tf.uint8)
    png = tf.io.encode_png(pred_uint8)
    out_path = os.path.join(preview_dir, f'pred_at_epoch_{epoch_idx+1:04d}.png')
    tf.io.write_file(out_path, png)
    return out_path

def generate_images(model, test_input, tar, epoch, checkpoint_dir, use_training_mode=True, seed=1234):
    # Ensure preview is reproducible if desired
    _set_all_seeds(seed)
    prediction = model(test_input, training=use_training_mode)

    # Save matplotlib side-by-side preview
    plt.figure(figsize=(15, 5))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    preview_dir = os.path.join(checkpoint_dir, 'previews')
    os.makedirs(preview_dir, exist_ok=True)
    plt.savefig(os.path.join(preview_dir, f'image_at_epoch_{epoch+1:04d}.png'), bbox_inches='tight')
    plt.close()

    # Also save the raw predicted image to disk for exact comparison
    raw_path = _save_raw_prediction(prediction, preview_dir, epoch)
    print(f"Saved preview figure and raw prediction: {raw_path}")

@tf.function
def train_step(input_image, target, generator, discriminator, gen_optimizer, disc_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)
        gen_total_loss, _, _ = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    return disc_loss, gen_total_loss

# --- 6. MAIN TRAINING FUNCTION ---
def fit(train_ds, test_ds, epochs, checkpoint_dir, preview_use_training_mode=True, preview_seed=1234):
    generator = Generator()
    discriminator = Discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    start_epoch = 0
    if latest_ckpt:
        print(f"✅ Restoring from checkpoint: {latest_ckpt}")
        checkpoint.restore(latest_ckpt)
        try:
            start_epoch = int(os.path.basename(latest_ckpt).split('-')[-1])
            print(f"🔄 Resuming training from epoch {start_epoch + 1}")
        except:
            print("Could not determine start epoch. Starting from 0.")
    else:
        print("🏁 No checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        # Generate preview (with optional training/eval mode + seed)
        for example_input, example_target in test_ds.take(1):
            generate_images(generator, example_input, example_target, epoch, checkpoint_dir,
                            use_training_mode=preview_use_training_mode, seed=preview_seed)

        print(f"Epoch: {epoch+1}/{epochs}")
        for n, (input_image, target) in train_ds.enumerate():
            disc_loss, gen_loss = train_step(input_image, target, generator, discriminator,
                                             generator_optimizer, discriminator_optimizer)
            if (n + 1) % 100 == 0:
                print('.', end='', flush=True)
        print()
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'Time for epoch {epoch + 1} is {time.time()-start_time:.2f} sec')
        print(f'  Checkpoint saved for epoch {epoch+1}')
        print(f'  Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}')

    print("🎉 Training complete!")

# --- 7. SCRIPT EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a pix2pix model for SAR image colorization.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the root of the dataset folder (e.g., ./v_2).')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--checkpoint_dir', type=str, default='./training_checkpoints',
                        help='Directory to save checkpoints.')
    parser.add_argument('--preview_use_training_mode', type=str, default='true',
                        help='true => preview with training=True (dropout/BN as in training); false => training=False')
    parser.add_argument('--preview_seed', type=int, default=1234,
                        help='Seed for preview generation to make it reproducible.')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Make ops deterministic as far as possible
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    # Load file paths
    all_s1_paths, all_s2_paths = load_image_paths(args.dataset_path)
    print(f"Found {len(all_s1_paths)} correctly paired images.")

    # Create the dataset pipeline
    dataset = tf.data.Dataset.from_tensor_slices((all_s1_paths, all_s2_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    # Use a small part of the dataset for generating previews
    test_dataset = dataset.take(5)

    # Start training
    fit(dataset, test_dataset, args.epochs, args.checkpoint_dir,
        preview_use_training_mode=(args.preview_use_training_mode.lower() == 'true'),
        preview_seed=args.preview_seed)
