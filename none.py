import numpy as np
import random
from collections import defaultdict
from atarigon.api import Goshi, Goban,Ten
from typing import Optional
import copy


class Noneplayer(Goshi):
    """The player claims the first empty position it finds."""

    def __init__(self):
        """Initializes the player."""
        super().__init__(f'Noneplayer')

    def decide(self, goban: 'Goban') -> Optional[Ten]:

        return None