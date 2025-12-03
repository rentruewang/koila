# Copyright (c) AIoWay Authors - All Rights Reserved

import gha
import pdm

if __name__ == "__main__":

    gha.setup()

    pdm.sync()

    pdm.run("pytest")
