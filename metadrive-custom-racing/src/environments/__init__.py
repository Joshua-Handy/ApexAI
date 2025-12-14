"""Environments package for metadrive-custom-racing."""

# Import modules with graceful error handling
try:
    from .single_car_racing import create_racing_environment
    __all__ = ["create_racing_environment"]
except ImportError as e:
    # If single_car_racing imports fail, still allow other environments to work
    __all__ = []
    pass
