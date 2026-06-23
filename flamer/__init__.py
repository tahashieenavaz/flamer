from importlib.metadata import version

__version__ = version("flamer")

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
from .gated_linear_units import SwiGLU
from .gated_linear_units import ReGLU
from .gated_linear_units import GeGLU
from .gated_linear_units import MiGLU
from .gated_linear_units import CeGLU
from .gated_linear_units import SeGLU
from .gated_linear_units import TaGLU
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

from .FeedForward import FeedForward
from .Reshape import Reshape
from .Permute import Permute

from .siren import Siren
from .siren import SirenActivation
