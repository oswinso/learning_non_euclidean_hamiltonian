from typing import Optional, Protocol, Tuple

import haiku as hk

from utils.types import ja


class IntegrableProtocol(Protocol):
    def d_R(self, params: hk.Params, R: ja, p: ja) -> ja:
        ...

    def d_Pi(self, params: hk.Params, R: ja, p: ja, u: Optional[ja] = None) -> ja:
        ...


class MultiIntegrableProtocol(Protocol):
    def d_q(self, params: hk.Params, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        ...

    def d_p(self, params: hk.Params, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        ...

    def d_R(self, params: hk.Params, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        ...

    def d_R_Pi1(self, params: hk.Params, q: ja, p: ja, R: ja, Pi: ja, h: float) -> Tuple[ja, ja]:
        ...

    def d_Pi1(self, params: hk.Params, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        ...

    def d_Pi2(self, params: hk.Params, q: ja, p: ja, R: ja, Pi: ja) -> ja:
        ...
