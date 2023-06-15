from typing import List, Tuple

from starlette.types import ASGIApp, Receive, Scope, Send

Headers = List[Tuple[bytes, bytes]]

# original developed by researchers at university of cambridge:
# https://pypi.org/project/fastapi-proxiedheadersmiddleware/

# modification to resolve Connection: keep-alive vs. upgrade conflict
# conflict described here: https://github.com/orgs/community/discussions/57596
# doesn't seem to work as intended, as the upgrade decision is made
# before the request goes through the middleware


class ProxiedHeadersMiddleware:
    """
    A middleware that modifies the request to ensure that FastAPI uses the
    X-Forwarded-* headers when creating URLs used to reference this application.

    We are very permissive in allowing all X-Forwarded-* headers to be used, as
    we know that this API will be published behind the API Gateway, and is
    therefore not prone to redirect hijacking.

    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope["headers"] = self.remap_headers(scope.get("headers", {}))

        await self.app(scope, receive, send)
        return

    def remap_headers(self, source: Headers) -> Headers:
        """
        Map X-Forwarded-Host to host and X-Forwarded-Prefix to prefix.

        """
        upgrade = len(
            [q for p, q in source if "connection" in str(p) and "upgrade" in str(q)]
        )

        source = dict(source)
        if upgrade:  # resolve conflict with priority of upgrade
            source[b"connection"] = b"upgrade"

        if b"x-forwarded-host" in source:
            source.update({b"host": source[b"x-forwarded-host"]})
            source.pop(b"x-forwarded-host")

        if b"x-forwarded-prefix" in source:
            source.update({b"host": source[b"host"] + source[b"x-forwarded-prefix"]})
            source.pop(b"x-forwarded-prefix")

        source = [(k, v) for k, v in source.items()]

        return source
