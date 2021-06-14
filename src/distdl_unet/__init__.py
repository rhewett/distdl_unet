__version__ = '0.0.0-dev'

from . import logging_timer  # noqa: F401
from .unet_base import UNetBase  # noqa: F401
from .unet_base import UNetLevelBase  # noqa: F401
from .unet_classic import ClassicalUNet  # noqa: F401
from .unet_dummy import DummyUNet  # noqa: F401

try:
	import distdl
	from .unet_distributed import DistributedUNet  # noqa: F401
except:
	# Don't die if DistDL is not installed
	pass
