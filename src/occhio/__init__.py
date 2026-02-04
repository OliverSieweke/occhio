# ABOUTME: Package entry exporting key occhio components.
# ABOUTME: Keeps minimal public interface for toy superposition tooling.
def hello():
    """Hello occhio"""
    print("Hello from occhio!!")

from .autoencoder import AutoEncoder

__all__ = ["hello", "AutoEncoder"]
