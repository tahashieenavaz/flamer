from .reinforcement_learning import HLGaussLoss
from .image import LayerNorm2d
from .image import SqueezeExcitation
from .image import PatchEmbedding
from .image import PatchEmbedding as PatchEmbed
from .loss import BalancedSoftmaxLoss
from .loss import LearnableLogitNormLoss
from .loss import LogitNormLoss
from .normalization import RMSNorm1d
from .stochastic import StochasticDepth
from .gated_linear_units import GatedLinearUnit
from .gated_linear_units import SwishGLU
from .gated_linear_units import ReLUGLU
from .gated_linear_units import GELUGLU
from .gated_linear_units import MishGLU
from .gated_linear_units import CELUGLU
from .gated_linear_units import SELUGLU
from .gated_linear_units import TanhGLU
from .channel_repository import AlexNetworkChannelRepository
from .channel_repository import ConvNextChannelRepository
from .channel_repository import DenseNetworkChannelRepository
from .channel_repository import EfficientNetworkChannelRepository
from .channel_repository import EfficientNetworkV2ChannelRepository
from .channel_repository import MaxVITChannelRepository
from .channel_repository import ResidualNetworkChannelRepository
from .channel_repository import SwinChannelRepository
from .decoy import PrintShape
from .decoy import PrintMean
from .decoy import PrintStd
from .siren import Siren
from .siren import SirenActivation
from .streams import Stream
from .streams import MaskedStream
from .streams import SoftMaskedStream
from .positional_encodings import SinusoidalPositionalEmbeddings
from .operations.reshape import Reshape
from .operations.permute import Permute

from importlib.metadata import version

try:
    __version__ = version("flamer")
except:
    __version__ = "development"
