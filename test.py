import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random

IMG_HEIGHT = 256
IMG_WIDTH = 256
OUTPUT_CHANNELS = 3

def _set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

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

def run_inference(args):
    # Determinism (best effort)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    _set_all_seeds(args.seed)

    # Create the Generator model instance
    generator = Generator()

    # Restore the latest checkpoint (only generator is needed)
    checkpoint = tf.train.Checkpoint(generator=generator)
    latest_ckpt = tf.train.latest_checkpoint(args.checkpoint_dir)
    if not latest_ckpt:
        print(f"🛑 Error: No checkpoint found in directory '{args.checkpoint_dir}'")
        print("Please make sure you have trained the model and the path is correct.")
        return
    print(f"✅ Restoring model from checkpoint: {latest_ckpt}")
    checkpoint.restore(latest_ckpt).expect_partial()

    # Load and preprocess the input SAR image
    try:
        input_image = tf.io.read_file(args.input_image)
        input_image = tf.io.decode_image(input_image, channels=3, expand_animations=False)
        input_image = tf.cast(input_image, tf.float32)
    except Exception as e:
        print(f"🛑 Error loading image file: {e}")
        return

    input_image_resized = tf.image.resize(input_image, [IMG_HEIGHT, IMG_WIDTH])
    input_image_normalized = (input_image_resized / 127.5) - 1.0
    input_batch = tf.expand_dims(input_image_normalized, 0)

    # Generate the colorized prediction
    mode = args.use_training_mode.lower() == 'true'
    print(f"🎨 Colorizing image... (generator training={mode}, seed={args.seed})")
    prediction = generator(input_batch, training=mode)

    # Post-process the output image for saving
    output_01 = (prediction[0] * 0.5 + 0.5)
    output_uint8 = tf.cast(tf.round(output_01 * 255.0), tf.uint8)

    # Define the output filename
    base_name = os.path.basename(args.input_image)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_colorized.png"  # save as PNG consistently

    # Save the output image (raw, no matplotlib)
    encoded_image = tf.io.encode_png(output_uint8)
    tf.io.write_file(output_filename, encoded_image)
    print(f"🎉 Success! Colorized image saved as: {output_filename}")

    # Optional side-by-side preview
    if args.save_preview.lower() == 'true':
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Input SAR Image")
        plt.imshow(tf.cast(input_image_resized, tf.uint8))
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Generated Color Image")
        plt.imshow(output_uint8.numpy())
        plt.axis('off')
        fig_name = f"{name}_preview.png"
        plt.savefig(fig_name, bbox_inches='tight')
        plt.close()
        print(f"🖼️ Preview saved as: {fig_name}")

# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize a SAR image using a trained pix2pix model.')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the training_checkpoints folder.')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input SAR image you want to test.')
    parser.add_argument('--use_training_mode', type=str, default='true',
                        help='true => generator(training=True) to match training preview; false => deterministic eval mode.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed so dropout results can match training preview.')
    parser.add_argument('--save_preview', type=str, default='true',
                        help='true => also save a side-by-side matplotlib preview.')
    args = parser.parse_args()
    run_inference(args)
