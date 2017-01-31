from .standard_ga import IndividualGA, StandardGA
from .binary_ga import BinaryGA
from .real_ga import RealGA
from .diffusion_ga import DiffusionGA
from .migration_ga import MigrationGA

__all__ = ['StandardGA', 'IndividualGA', 'BinaryGA', 'RealGA', 'DiffusionGA', 'MigrationGA']