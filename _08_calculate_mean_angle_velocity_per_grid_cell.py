# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""Gets and prints the spreadsheet's header columns

Parameters
----------
file_loc : str
    The file location of the spreadsheet
print_cols : bool, optional
    A flag used to print the columns to the console (default is False)

Returns
-------
list
    a list of strings representing the header columns
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut f�r Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# ===================================================

from pathlib import Path

import os
import timeit
import time


main_dir = Path(os.getcwd())
os.chdir(main_dir)


if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
