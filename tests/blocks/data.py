# Copyright (c) RenChu Wang - All Rights Reserved

from mimesis import Fieldset
from numpy import random
from pandas import DataFrame


def generate_dataframe(count: int):
    fs = Fieldset(i=count)
    return DataFrame(
        {
            "id": fs("increment"),
            "name": fs("person.full_name"),
            "address": fs("address"),
            "email": fs("person.email"),
            "city": fs("address.city"),
            "state": fs("address.state"),
            "date_time": fs("datetime.datetime"),
            "randomdata": random.randint(0, 100, count),
        }
    )
