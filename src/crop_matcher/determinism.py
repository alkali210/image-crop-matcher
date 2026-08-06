import cv2


DEFAULT_SEED = 20260730


def seed_opencv(seed: int) -> None:
    set_seed = getattr(cv2, "setRNGSeed", None)
    if callable(set_seed):
        signed_seed = ((seed + 2**31) % 2**32) - 2**31
        set_seed(signed_seed)
