from __future__ import absolute_import

__all__= ['RegException', 'InvalidInputTokenException', 'PatternMismatchException',
            'EmptyInputException']
        
class RegException(Exception):
    pass

class InvalidInputTokenException(Exception):
    '''
    This is an invalid token exception
    '''
    def __init__(self, index, message="Input token is invalid. Please correct the input string and try again."):
        self.index = index
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f'{self.index} -> {self.message}'

class PatternMismatchException(Exception):
    '''
    This is an pattern mismatch exception
    '''
    def __init__(self, index, message="Input Pattern mismatch. Please correct the input string and try again."):
        self.index = index
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f'{self.index} -> {self.message}'

class EmptyInputException(Exception):
    '''
    Raised when input is empty
    '''
    def __init__(self, index, message="Input is empty."):
        self.index = index
        self.message = message
        super().__init__(self.message)
    def __str__(self):
        return f'{self.index} -> {self.message}'