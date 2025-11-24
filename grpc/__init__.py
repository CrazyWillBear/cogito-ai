"""
gRPC server package for Cogito AI research assistant.
"""

from .server import CogitoServicer, serve

__all__ = ['CogitoServicer', 'serve']
