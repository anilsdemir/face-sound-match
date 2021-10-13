"""
Script contains exceptions that can be occur at runtime.
"""


class CustomException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)


class FaultyEnvironmentVariableValue(CustomException):
    pass


class MissingEnvironmentVariable(CustomException):
    pass


class MissingDirectoryError(CustomException):
    pass


class MissingArgumentValueError(CustomException):
    pass


class CudaIsNotAvailable(CustomException):
    pass