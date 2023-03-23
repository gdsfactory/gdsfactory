from gdsfactory.config import logger

logger.warning(
    "from gdsfactory.component_reference is deprecated and will be soon removed."
    "Use from gdsfactory.component import Instance instead"
)


__all__ = "ComponentReference"
