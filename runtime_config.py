import platform


def configure_tensorflow_runtime(tf):
    """Configure TensorFlow to prefer hardware acceleration with CPU fallback."""
    tf.config.set_soft_device_placement(True)

    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = []

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass

        try:
            details = tf.config.experimental.get_device_details(gpu)
        except ValueError:
            details = {}

        gpu_names.append(details.get("device_name") or gpu.name)

    system = platform.system()
    machine = platform.machine()
    accelerator = "GPU" if gpus else "CPU"

    note = "TensorFlow will run on CPU because no GPU device is visible."
    if gpus:
        note = "TensorFlow will prefer GPU and fall back to CPU for unsupported ops."
    elif system == "Darwin" and machine == "arm64":
        note = (
            "TensorFlow does not target Apple Neural Engine directly. "
            "Install tensorflow-metal to use the Apple GPU; otherwise CPU is used."
        )

    return {
        "accelerator": accelerator,
        "gpu_count": len(gpus),
        "gpu_names": gpu_names,
        "platform": f"{system} {machine}",
        "note": note,
    }
