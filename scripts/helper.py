import pandas as pd
import collections
from datetime import datetime


def get_max_week_from_weeks(weeks: pd.Series) -> int:
    """
    Determine the maximum week number in a dataset for a given year (usually 38 for a finished year)

    Args:
        weeks (pd.Series): A series containing unique week numbers.

    Returns:
        int: The maximum week number calculated from the input series.
    """
    if len(weeks.unique()) == 52:
        max_week = 38
    else:
        max_week = 38 + len(weeks.unique())
        if max_week > 52:
            max_week = max_week - 52

    return max_week


def get_weeks_list(weeknummer: int) -> list:
    """
    Generate a list of weeks based on a given week number. As weeks don't start with 1 but rather with 39/40, this
    function can be used in order to generate a sequence of week numbers ending at a certain point 'weeknummer'

    Args:
        weeknummer (int): The specific week number used as a reference.

    Returns:
        list: A list of target week numbers.
    """
    # Create target weeks
    weeks = []
    if int(weeknummer) > 38:
        weeks = weeks + [i for i in range(39, int(weeknummer) + 1)]
    elif int(weeknummer) < 39:
        weeks = weeks + [i for i in range(39, 53)]
        weeks = weeks + [i for i in range(1, int(weeknummer) + 1)]

    return weeks

def get_prediction_weeks_list(weeknummer: int) -> list[int]:
    """
    Generate a list of weeks still to come, starting from the given week number.
    Academic weeks typically start at 39 and wrap around after 52.

    Args:
        weeknummer (int): The current week number.

    Returns:
        list[int]: A list of week numbers still to come.
    """
    weeknummer = int(weeknummer) + 1

    if weeknummer >= 39:
        # From current week up to the end of the year
        weeks = list(range(weeknummer, 53))
    else:
        # From current week until week 38
        weeks = list(range(weeknummer, 39))

    return weeks



def get_max_week(predict_year, max_year, data, key):
    if predict_year == max_year:
        max_week = get_max_week_from_weeks(data[data[key] == predict_year]["Weeknummer"])
    else:
        if predict_year == 2021:
            max_week = 38
        else:
            max_week = get_max_week_from_weeks(data[data[key] == predict_year]["Weeknummer"])

    return max_week


def increment_week(week):
    if week == 52:
        return 1
    else:
        return week + 1


def decrement_week(week):
    if week == 1:
        return 52
    else:
        return week - 1


def convert_nan_to_zero(number):
    if pd.isnull(number):
        return 0
    else:
        return number


def get_all_weeks_valid(columns):
    return list(
        (collections.Counter(get_all_weeks_ordered()) & collections.Counter(columns)).elements()
    )


def get_all_weeks_ordered():
    return [
        "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "1", "2", "3", "4", "5", "6",
        "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26",
        "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38"
    ]


def get_pred_len(weeknummer: int) -> int:
    pred_len = (38 + 52 - weeknummer) if weeknummer > 38 else (38 - weeknummer)
    return pred_len

def is_current_week(year: int, week: int) -> bool:
    today = datetime.today()
    return year == today.year and week == today.isocalendar().week

def academic_week_number(year, week):
    """
    Convert (year, week) to a linear number where week 39 is treated as week 0
    and the year runs from week 39..52 + 1..38.
    """
    # Shift week numbers: 39→0, 40→1, …, 52→13, 1→14, …, 38→51
    if week >= 39:
        return (year, week - 39)
    else:
        return (year + 1, week + 13)  # push weeks 1–38 into the next academic year


