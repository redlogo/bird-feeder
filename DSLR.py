#!/usr/bin/env python

# Bird Feeder - Feed Birds & Capture Images!
# Copyright (C) 2020 redlogo
#
# This program is under MIT license

import argparse
import logging
import os
import gphoto2 as gp


def main():
    """
    main function interface
    :return: nothing
    """
    parser = argparse.ArgumentParser(description='parse the path and filename')
    parser.add_argument('--DSLRPICDIR', help="DSLR PIC DIR", required=True)
    parser.add_argument('--FILENAME', help="FILE NAME", required=True)
    args = parser.parse_args()

    # initialize DSLR_pics
    logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s', level=logging.WARNING)
    callback_obj = gp.check_result(gp.use_python_logging())
    DSLR = gp.Camera()
    DSLR.init()
    print('RPi Bird Feeder -> DSLR Ready')

    DSLR_path = DSLR.capture(gp.GP_CAPTURE_IMAGE)
    target = os.path.join(args.DSLRPICDIR, args.FILENAME)
    DSLR_file = DSLR.file_get(DSLR_path.folder, DSLR_path.name, gp.GP_FILE_TYPE_NORMAL)
    DSLR_file.save(target)
    DSLR.exit()


if __name__ == "__main__":
    main()
