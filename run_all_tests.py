import os
import logging

from domainadaptation.tests import test_discrepancy

if __name__ == "__main__":
    
    # disable useless tf warnings
    logging.disable(logging.WARNING)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    test_discrepancy()