import argparse
import os
import json
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "skin_model.keras")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Train the skin disease classifier.")
    parser.add_argument(
        "--data_dir",
        default="dataset/IMG_CLASSES",
        help="Path to the IMG_CLASSES dataset folder.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of initial training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop training after this many epochs without validation accuracy improvement.",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=5,
        help="Additional epochs for fine-tuning the top MobileNetV2 layers. Use 0 to skip.",
    )
    parser.add_argument(
        "--fine_tune_at",
        type=int,
        default=100,
        help="MobileNetV2 layer index where fine-tuning starts.",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
        help="Softens labels to reduce overconfidence and improve generalization.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2/AdamW regularization strength.",
    )
    return parser.parse_args()


def count_images_by_class(data_dir, class_names):
    image_extensions = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}
    counts = {}

    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        counts[class_name] = sum(
            1
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in image_extensions
        )

    return counts


def make_balanced_class_weights(class_counts, class_names):
    total_images = sum(class_counts.values())
    class_count = len(class_names)

    return {
        index: total_images / (class_count * class_counts[class_name])
        for index, class_name in enumerate(class_names)
        if class_counts[class_name] > 0
    }


def main():
    args = parse_args()

    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    from runtime_config import configure_tensorflow_runtime

    runtime_info = configure_tensorflow_runtime(tf)
    print(f"Runtime: {runtime_info['accelerator']} ({', '.join(runtime_info['gpu_names']) or 'CPU'})")

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds = image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        verbose=False,
    )

    val_ds = image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        verbose=False,
    )

    class_names = train_ds.class_names
    train_batches = tf.data.experimental.cardinality(train_ds).numpy()
    val_batches = tf.data.experimental.cardinality(val_ds).numpy()
    class_counts = count_images_by_class(args.data_dir, class_names)
    class_weights = make_balanced_class_weights(class_counts, class_names)
    class_weight_tensor = tf.constant(
        [class_weights[index] for index in range(len(class_names))],
        dtype=tf.float32,
    )

    with open(CLASS_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f)

    print(f"Classes: {', '.join(class_names)}")
    print(
        "Class counts: "
        + ", ".join(f"{name}={class_counts[name]}" for name in class_names)
    )
    print(
        "Balanced class weights: "
        + ", ".join(f"{name}={class_weights[index]:.3f}" for index, name in enumerate(class_names))
    )
    print(f"Batches: train={train_batches}, validation={val_batches}")

    def prepare_train_sample(image, label):
        one_hot_label = tf.one_hot(label, depth=len(class_names))
        sample_weight = tf.gather(class_weight_tensor, label)
        return image, one_hot_label, sample_weight

    def prepare_validation_sample(image, label):
        return image, tf.one_hot(label, depth=len(class_names))

    train_ds = train_ds.map(prepare_train_sample, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(prepare_validation_sample, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        data_augmentation,
        layers.Rescaling(1./127.5, offset=-1),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(
            len(class_names),
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(args.weight_decay),
        )
    ])

    def compile_model(learning_rate, total_epochs):
        steps_per_epoch = max(1, train_batches)
        decay_steps = max(1, steps_per_epoch * total_epochs)
        learning_rate_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=0.05,
        )
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate_schedule,
            weight_decay=args.weight_decay,
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=args.label_smoothing
            ),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
        )

    compile_model(learning_rate=1e-3, total_epochs=args.epochs)

    class CleanEpochLogger(tf.keras.callbacks.Callback):
        def __init__(self, phase, total_epochs):
            super().__init__()
            self.phase = phase
            self.total_epochs = total_epochs

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            print(
                f"{self.phase} epoch {epoch + 1}/{self.total_epochs}: "
                f"accuracy={logs.get('accuracy', 0):.4f}, "
                f"val_accuracy={logs.get('val_accuracy', 0):.4f}, "
                f"loss={logs.get('loss', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}"
            )

    def make_callbacks(phase, total_epochs, initial_value_threshold=None):
        checkpoint_options = {
            "filepath": MODEL_PATH,
            "monitor": "val_accuracy",
            "mode": "max",
            "save_best_only": True,
            "verbose": 1,
        }
        if initial_value_threshold is not None:
            checkpoint_options["initial_value_threshold"] = initial_value_threshold

        return [
            CleanEpochLogger(phase, total_epochs),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=args.patience,
                mode="max",
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(**checkpoint_options),
        ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=0,
        callbacks=make_callbacks("Initial training", args.epochs),
    )

    final_history = history.history
    best_val_accuracy = max(final_history["val_accuracy"])

    if args.fine_tune_epochs > 0:
        base_model.trainable = True

        for layer in base_model.layers[:args.fine_tune_at]:
            layer.trainable = False

        # Keep BatchNorm statistics stable while fine-tuning a small dataset.
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        compile_model(learning_rate=1e-5, total_epochs=args.fine_tune_epochs)

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            verbose=0,
            callbacks=make_callbacks(
                "Fine-tuning",
                args.fine_tune_epochs,
                initial_value_threshold=best_val_accuracy,
            ),
        )
        fine_tune_best_val_accuracy = max(fine_tune_history.history["val_accuracy"])

        if fine_tune_best_val_accuracy >= best_val_accuracy:
            final_history = fine_tune_history.history
            best_val_accuracy = fine_tune_best_val_accuracy
        else:
            print(
                "Fine-tuning did not beat the initial validation accuracy; "
                "keeping the best initial model."
            )
            model = tf.keras.models.load_model(MODEL_PATH)

    model.save(MODEL_PATH)

    print("Training completed.")
    print(
        "Final metrics: "
        f"accuracy={final_history['accuracy'][-1]:.4f}, "
        f"loss={final_history['loss'][-1]:.4f}, "
        f"val_accuracy={final_history['val_accuracy'][-1]:.4f}, "
        f"val_loss={final_history['val_loss'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
