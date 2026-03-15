try:
    from src.modules.losses.contperceptual import (
        LPIPSWithDiscriminator,
        MSEWithDiscriminator,
        LPIPSWithDiscriminator3D,
    )
except ImportError:
    pass
