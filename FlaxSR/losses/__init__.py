from .pixel_wise_losses import (
    l1_loss as l1_loss,
    l2_loss as l2_loss,
    charbonnier_loss as charbonnier_loss,
)
from .perceptual_loss import (
    vgg_loss as vgg_loss,
    check_vgg_params_exists as check_vgg_params_exists,
)


check_vgg_params_exists()
