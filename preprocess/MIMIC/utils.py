import json
import os

import pandas as pd
import numpy as np
import math
import psutil
from loguru import logger


def get_memory_info():
    pid = os.getpid()
    process = psutil.Process(pid)
    memory_info = process.memory_info().rss
    memory_info_mb = memory_info / 1024 / 1024


def dataframe_from_csv(path, header=0, index_col=0):
    return pd.read_csv(
        path,
        header=header,
        index_col=index_col,
        encoding="ISO-8859-1",
        low_memory=False,
    )


def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, "PATIENTS.csv"))
    pats = pats[["SUBJECT_ID", "GENDER", "DOB", "DOD"]]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats


def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, "ADMISSIONS.csv"))
    admits = admits[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "DEATHTIME",
            "ETHNICITY",
            "DIAGNOSIS",
        ]
    ]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, "ICUSTAYS.csv"))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays


def add_inhospital_mortality_to_icustays(stays, admissions):
    stays = stays.merge(admissions, left_on="HADM_ID", right_on="HADM_ID", how="inner")
    mortality = stays.DEATHTIME.notnull() & (
        (stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)
    )
    stays["MORTALITY"] = mortality.astype(int)
    stays = stays[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "DBSOURCE",
            "FIRST_CAREUNIT",
            "LAST_CAREUNIT",
            "FIRST_WARDID",
            "LAST_WARDID",
            "INTIME",
            "OUTTIME",
            "LOS",
            "MORTALITY",
        ]
    ]
    return stays


def read_icd_diagnoses_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, "D_ICD_DIAGNOSES.csv"))
    codes = codes[["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"]]
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, "DIAGNOSES_ICD.csv"))
    diagnoses = diagnoses.merge(
        codes, how="inner", left_on="ICD9_CODE", right_on="ICD9_CODE"
    )
    diagnoses[["SUBJECT_ID", "HADM_ID", "SEQ_NUM"]] = diagnoses[
        ["SUBJECT_ID", "HADM_ID", "SEQ_NUM"]
    ].astype(int)
    return diagnoses


diagnosis_labels = [
    "4019",
    "4280",
    "41401",
    "42731",
    "25000",
    "5849",
    "2724",
    "51881",
    "53081",
    "5990",
    "2720",
    "2859",
    "2449",
    "486",
    "2762",
    "2851",
    "496",
    "V5861",
    "99592",
    "311",
    "0389",
    "5859",
    "5070",
    "40390",
    "3051",
    "412",
    "V4581",
    "2761",
    "41071",
    "2875",
    "4240",
    "V1582",
    "V4582",
    "V5867",
    "4241",
    "40391",
    "78552",
    "5119",
    "42789",
    "32723",
    "49390",
    "9971",
    "2767",
    "2760",
    "2749",
    "4168",
    "5180",
    "45829",
    "4589",
    "73300",
    "5845",
    "78039",
    "5856",
    "4271",
    "4254",
    "4111",
    "V1251",
    "30000",
    "3572",
    "60000",
    "27800",
    "41400",
    "2768",
    "4439",
    "27651",
    "V4501",
    "27652",
    "99811",
    "431",
    "28521",
    "2930",
    "7907",
    "E8798",
    "5789",
    "79902",
    "V4986",
    "V103",
    "42832",
    "E8788",
    "00845",
    "5715",
    "99591",
    "07054",
    "42833",
    "4275",
    "49121",
    "V1046",
    "2948",
    "70703",
    "2809",
    "5712",
    "27801",
    "42732",
    "99812",
    "4139",
    "3004",
    "2639",
    "42822",
    "25060",
    "V1254",
    "42823",
    "28529",
    "E8782",
    "30500",
    "78791",
    "78551",
    "E8889",
    "78820",
    "34590",
    "2800",
    "99859",
    "V667",
    "E8497",
    "79092",
    "5723",
    "3485",
    "5601",
    "25040",
    "570",
    "71590",
    "2869",
    "2763",
    "5770",
    "V5865",
    "99662",
    "28860",
    "36201",
    "56210",
]


def extract_diagnosis_labels(diagnoses):
    global diagnosis_labels
    diagnoses["VALUE"] = 1
    labels = (
        diagnoses[["HADM_ID", "ICD9_CODE", "VALUE"]]
        .drop_duplicates()
        .pivot(index="HADM_ID", columns="ICD9_CODE", values="VALUE")
        .fillna(0)
        .astype(int)
    )
    for l in diagnosis_labels:
        if l not in labels:
            labels[l] = 0
    labels = labels[diagnosis_labels]
    return labels.rename(
        dict(zip(diagnosis_labels, ["Diagnosis " + d for d in diagnosis_labels])),
        axis=1,
    )


def read_events_table(mimic3_path, var_map):
    # events = dataframe_from_csv(os.path.join(mimic3_path, 'CHARTEVENTS.csv'))
    concatenated_chunks = []
    for chunk in pd.read_csv(
        os.path.join(mimic3_path, "CHARTEVENTS.csv"),
        header=0,
        index_col=0,
        encoding="ISO-8859-1",
        chunksize=100000,
        low_memory=False,
    ):
        merged_chunk = chunk.merge(var_map, left_on="ITEMID", right_index=True)
        concatenated_chunks.append(merged_chunk)
    events = pd.concat(concatenated_chunks, ignore_index=True)
    events = events[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "CHARTTIME",
            "ITEMID",
            "VALUE",
            "VALUEUOM",
            "VARIABLE",
            "MIMIC_LABEL",
        ]
    ]
    events = events[events.VALUE.notnull()]
    events.CHARTTIME = pd.to_datetime(events.CHARTTIME)
    events.HADM_ID = events.HADM_ID.fillna(value=-1).astype(int)
    events.ICUSTAY_ID = events.ICUSTAY_ID.fillna(value=-1).astype(int)
    events.VALUEUOM = events.VALUEUOM.fillna("").astype(str)
    return events


def read_itemid_to_variable_map(fn, variable_column="LEVEL2"):
    var_map = dataframe_from_csv(fn, index_col=None).fillna("").astype(str)
    var_map.COUNT = var_map.COUNT.astype(int)
    var_map = var_map[(var_map[variable_column] != "") & (var_map.COUNT > 0)]
    var_map = var_map[(var_map.STATUS == "ready")]
    var_map.ITEMID = var_map.ITEMID.astype(int)
    var_map = var_map[[variable_column, "ITEMID", "MIMIC LABEL"]].set_index("ITEMID")
    return var_map.rename(
        {variable_column: "VARIABLE", "MIMIC LABEL": "MIMIC_LABEL"}, axis=1
    )


def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on="ITEMID", right_index=True)


def convert_events_to_timeseries(events, variable_column="VARIABLE", variables=[]):
    metadata = (
        events[["CHARTTIME", "ICUSTAY_ID"]]
        .sort_values(by=["CHARTTIME", "ICUSTAY_ID"])
        .drop_duplicates(keep="first")
        .set_index("CHARTTIME")
    )
    timeseries = (
        events[["CHARTTIME", variable_column, "VALUE"]]
        .sort_values(by=["CHARTTIME", variable_column, "VALUE"], axis=0)
        .drop_duplicates(subset=["CHARTTIME", variable_column], keep="last")
    )
    timeseries = (
        timeseries.pivot(index="CHARTTIME", columns=variable_column, values="VALUE")
        .merge(metadata, left_index=True, right_index=True)
        .sort_index(axis=0)
        .reset_index()
    )
    for v in variables:
        if v not in timeseries:
            timeseries[v] = np.nan
    return timeseries


def add_hours_elpased_to_events(events, dt, remove_charttime=True):
    events = events.copy()
    events["HOURS"] = (
        (events.CHARTTIME - dt).apply(lambda s: s / np.timedelta64(1, "s")) / 60.0 / 60
    )
    if remove_charttime:
        del events["CHARTTIME"]
    return events


def get_events_for_stay(events, icustayid, intime=None, outtime=None):
    idx = events.ICUSTAY_ID == icustayid
    if intime is not None and outtime is not None:
        idx = idx | ((events.CHARTTIME >= intime) & (events.CHARTTIME <= outtime))
    events = events[idx]
    return events


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_to_dict(data, channel_info):
    header = [
        "Hours",
        "Capillary refill rate",
        "Diastolic blood pressure",
        "Fraction inspired oxygen",
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
        "Glucose",
        "Heart Rate",
        "Height",
        "Mean blood pressure",
        "Oxygen saturation",
        "Respiratory rate",
        "Systolic blood pressure",
        "Temperature",
        "Weight",
        "pH",
    ]
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i - 1] = [
            (t, x)
            for (t, x) in zip(data[:, 0], data[:, i])
            if isinstance(x, str) and x != "nan"
        ]
        channel = header[i]
        if len(channel_info[channel]["possible_values"]) != 0:
            ret[i - 1] = list(
                map(lambda x: (x[0], channel_info[channel]["values"][x[1]]), ret[i - 1])
            )
        ret[i - 1] = list(
            map(
                lambda x: (float(x[0]), float(x[1]) if is_number(x[1]) else np.nan),
                ret[i - 1],
            )
        )
    return ret


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_results(names, pred, y_true, path):
    create_directory(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("stay,prediction,y_true\n")
        for name, x, y in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))


def get_first_valid_from_timeseries(timeseries, variable):
    if variable in timeseries:
        idx = timeseries[variable].notnull()
        if idx.any():
            loc = np.where(idx)[0][0]
            return timeseries[variable].iloc[loc]
    return np.nan
