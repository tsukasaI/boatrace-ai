"""
Custom exceptions for boatrace-ai
"""


class BoatraceError(Exception):
    """Base exception for boatrace-ai"""
    pass


class DataError(BoatraceError):
    """Error related to data loading or processing"""
    pass


class ModelError(BoatraceError):
    """Error related to model operations"""
    pass


class PredictionError(BoatraceError):
    """Error related to predictions"""
    pass


class ValidationError(BoatraceError):
    """Error related to input validation"""
    pass


class ConfigurationError(BoatraceError):
    """Error related to configuration"""
    pass
