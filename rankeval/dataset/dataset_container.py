# Copyright (c) 2017, All Contributors (see CONTRIBUTORS file)
# Authors: Franco Maria Nardini <francomaria.nardini@isti.cnr.it>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

class DatasetContainer(object):
    """
    This class is a container used to easily manage a dataset and associated
    learning to rank models trained by using it. It also offers the possibility
    to store the license coming with public dataset.
    """
    train_dataset = None
    validation_dataset = None
    test_dataset = None

    license_agreement = ''

    model_filenames = None