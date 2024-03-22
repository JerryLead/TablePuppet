import numpy as np
import pandas as pd
from scipy.stats import skew
import json
import os
from loguru import logger


import utils
import cleaner
import feature_extractor

data_path = "/datadisk/xxx/mimic3/physionet.org/files/mimiciii/1.4/data"

patients = utils.read_patients_table(data_path)

logger.info("Finish loading patients table.")
utils.get_memory_info()

admissions = utils.read_admissions_table(data_path).drop("SUBJECT_ID", axis=1)
logger.info("Finish loading admissions table.")
utils.get_memory_info()

stays = utils.read_icustays_table(data_path)
stays = utils.add_inhospital_mortality_to_icustays(stays, admissions)
logger.info("Finish loading stays table.")
utils.get_memory_info()

diagnoses = utils.read_icd_diagnoses_table(data_path)
diagnoses = utils.extract_diagnosis_labels(diagnoses)
logger.info("Finish loading diagnoses table.")
utils.get_memory_info()

var_map = utils.read_itemid_to_variable_map("itemid_to_variable_map.csv")
events = utils.read_events_table(data_path, var_map)
logger.info("Finish loading events table.")
utils.get_memory_info()
# events = utils.map_itemids_to_variables(events, var_map)

events = cleaner.clean_events(events)
logger.info("Finish cleaning events table.")
utils.get_memory_info()

timeseries = utils.convert_events_to_timeseries(
    events, variables=var_map.VARIABLE.unique()
)
logger.info("Finish converting events to timeseries.")
utils.get_memory_info()

del events
logger.info("Finish deleting events table.")
utils.get_memory_info()

episodes = None
for i in range(stays.shape[0]):
    stay_id = stays.ICUSTAY_ID.iloc[i]
    intime = stays.INTIME.iloc[i]
    outtime = stays.OUTTIME.iloc[i]
    episode = utils.get_events_for_stay(timeseries, stay_id, intime, outtime)
    episode = utils.add_hours_elpased_to_events(episode, intime).sort_index(axis=0)
    if stay_id in episode.index:
        episode.loc[stay_id, "Weight"] = utils.get_first_valid_from_timeseries(
            episode, "Weight"
        )
        episode.loc[stay_id, "Height"] = utils.get_first_valid_from_timeseries(
            episode, "Height"
        )
    columns = list(episode.columns)
    columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
    episode = episode[columns_sorted]
    if episodes is None:
        episodes = [episode]
    else:
        episodes.append(episode)

del timeseries
logger.info("Delete timeseries.")
utils.get_memory_info()

episodes = pd.concat(episodes)
eps = 1e-6
episodes = episodes.loc[episodes["HOURS"].apply(lambda x: -eps <= x <= 48 + eps)]
logger.info("Finish getting episodes.")
utils.get_memory_info()

records = episodes.to_dict("records")
logger.info("Finish converting episodes to dict.")
utils.get_memory_info()

del episodes
logger.info("Delete episodes.")
utils.get_memory_info()

icustay_measures = {}
for record in records:
    icu_stay_id = record["ICUSTAY_ID"]
    item = [
        record["HOURS"],
        record["Capillary refill rate"],
        record["Diastolic blood pressure"],
        record["Fraction inspired oxygen"],
        record["Glascow coma scale eye opening"],
        record["Glascow coma scale motor response"],
        record["Glascow coma scale total"],
        record["Glascow coma scale verbal response"],
        record["Glucose"],
        record["Heart Rate"],
        record["Height"],
        record["Mean blood pressure"],
        record["Oxygen saturation"],
        record["Respiratory rate"],
        record["Systolic blood pressure"],
        record["Temperature"],
        record["Weight"],
        record["pH"],
    ]
    if icu_stay_id not in icustay_measures:
        icustay_measures[icu_stay_id] = [item]
    else:
        icustay_measures[icu_stay_id].append(item)
del records
logger.info("Delete records.")
utils.get_memory_info()

for icustay_id, measures in icustay_measures.items():
    icustay_measures[icustay_id] = np.array(measures, dtype=str)
logger.info("Finish converting measures to np.array.")
utils.get_memory_info()

with open("channel_info.json") as channel_info_file:
    channel_info = json.loads(channel_info_file.read())

logger.info("Begin converting to dict.")
data = {}
for icustay_id, measures in icustay_measures.items():
    data[icustay_id] = utils.convert_to_dict(measures, channel_info)
logger.info("Finish converting to dict.")
utils.get_memory_info()

all_functions = [min, max, np.mean, np.std, skew, len]
functions_map = {"all": all_functions, "len": [len], "all_but_len": all_functions[:-1]}
periods_map = {
    "all": (0, 0, 1, 0),
    "first4days": (0, 0, 0, 4 * 24),
    "first8days": (0, 0, 0, 8 * 24),
    "last12hours": (1, -12, 1, 0),
    "first25percent": (2, 25),
    "first50percent": (2, 50),
}
features = {}
logger.info("Begin extracting features.")
for icustay_id, measures in data.items():
    features[icustay_id] = feature_extractor.extract_features_single_episode(
        measures, periods_map["all"], functions_map["all"]
    )
logger.info("Finish extracting features.")
utils.get_memory_info()
features = pd.DataFrame.from_dict(features, orient="index")
features.rename_axis("ICUSTAY_ID", axis="index", inplace=True)

logger.info("Begin saving tables to disk.")
patients.to_csv(os.path.join("target", "patients.csv"), index=False)
admissions.to_csv(os.path.join("target", "admissions.csv"), index=False)
diagnoses.to_csv(os.path.join("target", "diagnoses.csv"), index=True)
stays.to_csv(os.path.join("target", "stays.csv"), index=False)
features.to_csv(os.path.join("target", "events.csv"), index=True)
logger.info("Finished sucessfully!")
