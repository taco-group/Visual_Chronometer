# Loss modules are only needed for training; skip import errors at inference time.
try:
    from src.modules.losses.contperceptual import (
        LPIPSWithDiscriminator,
        MSEWithDiscriminator,
        LPIPSWithDiscriminator3D,
    )
except ImportError:
    pass
