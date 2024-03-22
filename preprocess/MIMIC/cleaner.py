import numpy as np
from pandas import Series
import re


# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match("^(\d+)/(\d+)$", s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.VALUE.astype(str).copy()
    idx = v.apply(lambda s: "/" in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match("^(\d+)/(\d+)$", s).group(2))
    return v.astype(float)


# CRR: strings with brisk, <3 normal, delayed, or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan

    # when df.VALUE is empty, dtype can be float and comparision with string
    # raises an exception, to fix this we change dtype to str
    df_value_str = df.VALUE.astype(str)

    v.loc[(df_value_str == "Normal <3 secs") | (df_value_str == "Brisk")] = 0
    v.loc[(df_value_str == "Abnormal >3 secs") | (df_value_str == "Delayed")] = 1
    return v


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.VALUE.astype(float).copy()

    """ The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    """
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    """ The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    """
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE > 1.0)

    """ The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    """
    is_str = np.array(map(lambda x: type(x) == str, list(df.VALUE)), dtype=np.bool)
    idx = df.VALUEUOM.fillna("").apply(lambda s: "torr" not in s.lower()) & (
        is_str | (~is_str & (v > 1.0))
    )

    v.loc[idx] = v[idx] / 100.0
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match("^(\d+(\.\d*)?|\.\d+)$", s))
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.VALUE.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match("^(\d+(\.\d*)?|\.\d+)$", s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = v <= 1
    v.loc[idx] = v[idx] * 100.0
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.VALUE.astype(float).copy()
    idx = (
        df.VALUEUOM.fillna("").apply(lambda s: "F" in s.lower())
        | df.MIMIC_LABEL.apply(lambda s: "F" in s.lower())
        | (v >= 79)
    )
    v.loc[idx] = (v[idx] - 32) * 5.0 / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.VALUE.astype(float).copy()
    # ounces
    idx = df.VALUEUOM.fillna("").apply(
        lambda s: "oz" in s.lower()
    ) | df.MIMIC_LABEL.apply(lambda s: "oz" in s.lower())
    v.loc[idx] = v[idx] / 16.0
    # pounds
    idx = (
        idx
        | df.VALUEUOM.fillna("").apply(lambda s: "lb" in s.lower())
        | df.MIMIC_LABEL.apply(lambda s: "lb" in s.lower())
    )
    v.loc[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.VALUE.astype(float).copy()
    idx = df.VALUEUOM.fillna("").apply(
        lambda s: "in" in s.lower()
    ) | df.MIMIC_LABEL.apply(lambda s: "in" in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


# ETCO2: haven't found yet
# Urine output: ambiguous units (raw ccs, ccs/kg/hr, 24-hr, etc.)
# Tidal volume: tried to substitute for ETCO2 but units are ambiguous
# Glascow coma scale eye opening
# Glascow coma scale motor response
# Glascow coma scale total
# Glascow coma scale verbal response
# Heart Rate
# Respiratory rate
# Mean blood pressure
clean_fns = {
    "Capillary refill rate": clean_crr,
    "Diastolic blood pressure": clean_dbp,
    "Systolic blood pressure": clean_sbp,
    "Fraction inspired oxygen": clean_fio2,
    "Oxygen saturation": clean_o2sat,
    "Glucose": clean_lab,
    "pH": clean_lab,
    "Temperature": clean_temperature,
    "Weight": clean_weight,
    "Height": clean_height,
}


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = events.VARIABLE == var_name
        try:
            events.loc[idx, "VALUE"] = clean_fn(events[idx])
        except Exception as e:
            import traceback

            print("Exception in clean_events:", clean_fn.__name__, e)
            print(traceback.format_exc())
            print("number of rows:", np.sum(idx))
            print("values:", events[idx])
            exit()
    return events.loc[events.VALUE.notnull()]
