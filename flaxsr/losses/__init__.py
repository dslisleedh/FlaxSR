from .pixel_wise_losses import (
    l1_loss as l1_loss,
    L1Loss as L1Loss,
    l2_loss as l2_loss,
    L2Loss as L2Loss,
    charbonnier_loss as charbonnier_loss,
    CharbonnierLoss as CharbonnierLoss,
)
from .perceptual_losses import (
    vgg_loss as vgg_loss,
    VGGLoss as VGGLoss,
)
from .adversarial_losses import (
    minmax_discriminator_loss as minmax_discriminator_loss,
    minmax_generator_loss as minmax_generator_loss,
    MinmaxDiscriminatorLoss as MinmaxDiscriminatorLoss,
    MinmaxGeneratorLoss as MinmaxGeneratorLoss,
    least_square_discriminator_loss as least_square_discriminator_loss,
    least_square_generator_loss as least_square_generator_loss,
    LeastSquareDiscriminatorLoss as LeastSquareDiscriminatorLoss,
    LeastSquareGeneratorLoss as LeastSquareGeneratorLoss,
    relativistic_discriminator_loss as relativistic_discriminator_loss,
    relativistic_generator_loss as relativistic_generator_loss,
    RelativisticDiscriminatorLoss as RelativisticDiscriminatorLoss,
    RelativisticGeneratorLoss as RelativisticGeneratorLoss,
)
from .utils import (
    check_vgg_params_exists as check_vgg_params_exists,
    load_vgg19_params as load_vgg19_params,
    # get_loss_wrapper as get_loss_wrapper,
    # compute_loss as compute_loss,
    Reduce as Reduces,
    LossWrapper as LossWrapper,
)


check_vgg_params_exists()
del check_vgg_params_exists
