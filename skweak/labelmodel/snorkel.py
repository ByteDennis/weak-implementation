from ..constant import ABSTAIN
from ..core import registry
from wrench.labelmodel import snorkel as SN

SN.ABSTAIN = ABSTAIN


@registry('label', aliases=["SN", "SNORKEL"])
class Snorkel(SN.Snorkel): ...

