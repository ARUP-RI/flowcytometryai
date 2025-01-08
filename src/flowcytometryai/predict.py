import hashlib
import os
import pickle
from dataclasses import dataclass
from flowcytometryai import logger

import numpy as np
import pandas as pd

# from flomo import logger, config
from flowcytometryai import PANEL_CONFIG
from flowcytometryai.data import Prediction, PredictionMetadata


pd.set_option("display.max_rows", 100)


def filter_multi(local_b_tube, local_t_tube, local_mye_tube):
    filtered_tubes = []
    for tube, tube_type in zip((local_b_tube, local_t_tube, local_mye_tube), ("B_cell", "T_cell", "Myeloid")):
        if "event_filter_keep" in tube.columns:
            logger.info("Processed tube file {tube_type} has event_filter_keep column and therefore unfiltered")
            logger.info("Filtering")
            filtered_tubes.append(tube[tube.event_filter_keep])
        else:
            logger.info("Processed tube file {tube_type} has no event_filter_keep column and therefore prefiltered")
            filtered_tubes.append(tube)
    return filtered_tubes


async def normalize1case(
    accession,
    b_cell_tube,
    t_cell_tube,
    mye_tube,
    b_cell_cytometer_id,
    t_cell_cytometer_id,
    mye_cytometer_id,
    b_cell_run_date,
    t_cell_run_date,
    mye_run_date,
    normalization_sample_count,
    df_metrics,
):
    """Given an instrument and a run date, get the last n samples for the instrument prior to the run date.
    25 column means and column std deviations from these will be used.
    z normalization - Subtract means and divide by std deviation for each of our input columns.
    13 columns of parameter data inside tube files. e.g.

    """
    tubes = [b_cell_tube, t_cell_tube, mye_tube]
    if not normalization_sample_count:
        return tubes

    tube_types = ("B_cell", "T_cell", "Myeloid")

    tube_params = (
        PANEL_CONFIG["Panels"]["B_cell"],
        PANEL_CONFIG["Panels"]["T_cell"],
        PANEL_CONFIG["Panels"]["Myeloid"],
    )
    instruments = (b_cell_cytometer_id, t_cell_cytometer_id, mye_cytometer_id)
    run_dates = (b_cell_run_date, t_cell_run_date, mye_run_date)
    df_suffixes = ["b", "t", "m"]
    normalized_tubes = []
    for tube_norm, tube_type, params, instrument, run_date, suffix in zip(
        tubes, tube_types, tube_params, instruments, run_dates, df_suffixes
    ):
        channel_vals = df_metrics.filter(regex="_me_unfiltered$|_sd_unfiltered$", axis=1)
        meds = channel_vals.median()
        logger.info(f"Normalization samples for sample {accession=}, {tube_type=}, {instrument=}, {run_date=}")
        logger.info(df_metrics[["accession", "guid"]].to_dict(orient="list"))
        logger.info(f"Normalization median values for sample {accession=}, {tube_type=}, {instrument=}, {run_date=}")
        logger.info(dict(meds))

        for param in params:
            mean_med = tube_norm[param].median()
            sd_med = tube_norm[param].std()
            tube_norm[param] = (tube_norm[param] - mean_med) / sd_med

        logger.info(
            f"Normalized first 3 rows of channel values for sample {accession=}, "
            f"{tube_type=}, {instrument=}, {run_date=}"
        )
        # Display the first 3 rows of the DataFrame with all columns
        pd.set_option("display.max_columns", None)
        logger.info(tube_norm.head(3))
        pd.reset_option("display.max_columns")

        logger.info(f"Normalized channel mean values for sample {accession=}, {tube_type=}, {instrument=}, {run_date=}")
        logger.info(dict(tube_norm.mean()))
        logger.info(f"Normalized channel std values for sample {accession=}, {tube_type=}, {instrument=}, {run_date=}")
        logger.info(dict(tube_norm.std()))

        normalized_tubes.append(tube_norm)

    return normalized_tubes  # list of [df_norm_b, df_norm_t, df_norm_m]


def project1case(
    input_tubes,
    centroid_paths,
    output_features,
    diagnoses,
    tube_names,
    som_dim,
    normalize_soms,
    num_workers,
):
    # make 3 projections = 1 SOM projections x 3 tube types
    projections = project_multi(
        centroid_paths,
        input_tubes,
        som_dim=som_dim,
        diagnoses=diagnoses,
        tube_names=tube_names,
        norm_soms=normalize_soms,
        num_workers=num_workers,
    )

    # order projections to match training order of model
    ordered = []
    for diag in diagnoses:
        for tube_name in tube_names:
            ordered.append(projections[tube_name][diag])
    # finally, vectorize combined projections
    features = np.concatenate(ordered).flatten()
    if output_features:
        np.save(output_features, features)

    logger.info("First 32 som projected features for sample")
    logger.info(features[:32])
    logger.info("Last 32 som projected features for sample")
    logger.info(features[-32:])
    logger.info(f"Greater than zero features: {features[features > 0]}")
    logger.info(f"Sum of features. {features.sum()}")
    return features


def project_multi(
    centroid_files,
    tubes,
    som_dim,
    diagnoses,
    tube_names,
    norm_soms,
    num_workers,
):
    """Make som projections needed for one case (one som per tube)

    :param tube_files:
    :param normed:
    :return:
    """
    projections = {}
    for tube_name, tube, centroid_file in zip(tube_names, tubes, centroid_files):
        # centroid_file = [os.path.join(centroid_dir, f"centroids_{tube_name}_{diag}.npy") for diag in diagnoses]
        projections[tube_name] = {}
        # assume only one centroid file and one diagnosis for now
        if len([centroid_file]) != 1:
            raise ValueError(f"Expected one centroid file but got {len(centroid_file)}")
        if len(diagnoses) != 1:
            raise ValueError(f"Expected one diagnosis but got {len(diagnoses)}")

        # for centroid, diag in zip([centroid_file], diagnoses):
        logger.info(f"Working on {centroid_file}")
        diag = diagnoses[0]
        projections[tube_name][diag] = project_one(
            centroid_file=centroid_file,
            tube=tube,
            som_dim=som_dim,
            norm_soms=norm_soms,
            num_workers=num_workers,
        )
    return projections


def project_one(centroid_file, tube, som_dim, norm_soms, num_workers):
    """Make single 2d histogram (32x32), SOM projection given SOM object and events array
    :param centroid_file:
    :param tube:
    :param som_dim:
    :param norm_soms:
    :param num_workers:

    :return: 32x32
    """
    import jax
    from flowanalysis.notorch_somax import SOM

    DEVICE_JAX = jax.devices()[0]

    centroids = jax.device_put(np.load(centroid_file))
    # dim should match the number of input features.
    som = SOM(n=som_dim, m=som_dim, dim=13, centroids=centroids, device=DEVICE_JAX)
    bmus, *_ = som.predict(
        tube.to_numpy(),
        num_workers=num_workers,
        batch_size=10000,
    )
    # make histogram-like projection on output space
    out_dim1, out_dim2 = som.m, som.n
    dim1_data = bmus[:, 0]
    dim2_data = bmus[:, 1]
    projection, xi, yi = np.histogram2d(
        dim1_data,
        dim2_data,
        bins=(range(out_dim1 + 1), range(out_dim2 + 1)),
        density=norm_soms,
    )
    np.set_printoptions(threshold=100)
    return projection


def predict1case(features, model, feat_indexes, som_dim, tube_names, diagnoses, threshold, scale_range):
    logger.info(f"Loading input features from file {features}")
    logger.info(f"Using trained model type {type(model).__name__}")
    # Get features and make sure they are right shape for model
    if len(features.shape) != 1:
        raise ValueError(f"Input feature set needs to be vector with only one dimension but had shape {features.shape}")
    expected_feats_count = len(diagnoses) * len(tube_names) * som_dim * som_dim
    if features.shape[0] != expected_feats_count:
        raise ValueError(
            f"Input feature set needs diagnosis count * tube count * SOM width * SOM height {expected_feats_count} "
            f"but input feature len was {features.shape}"
        )
    if feat_indexes:
        if max(feat_indexes) >= features.shape[0]:
            raise ValueError(
                f"feat_indexes max value {max(feat_indexes)} cannot extends past length of "
                f"input features {features.shape[0]}"
            )
        feat_subset = features[feat_indexes]
    else:
        feat_subset = features
    logger.info(f"np sum of feats = {np.sum(feat_subset)}")

    # run model
    try:
        logger.info("Trying model.predict_proba")
        res = model.predict_proba([feat_subset])
        pos_p_val_unscaled = res[0][1]
    except Exception as e:
        logger.info(f"Exception: {e}")
        logger.info("Trying model.predict")

        res = model.predict([feat_subset])
        pos_p_val_unscaled = res[0]

    logger.info(f"Prediction result unscaled = {pos_p_val_unscaled}")
    if scale_range is not None:
        pos_p_val = (pos_p_val_unscaled - scale_range[0]) / (scale_range[1] - scale_range[0])
        pos_p_val = 1.0 if pos_p_val > 1.0 else 0.0 if pos_p_val < 0.0 else pos_p_val
        logger.info(f"Prediction result scaled to range of {scale_range}")
    else:
        pos_p_val = pos_p_val_unscaled
        logger.info("Prediction result not scaled")
    logger.info(f"Final prediction result = {pos_p_val}")

    prediction = pos_p_val >= threshold
    return dict(
        prediction=bool(prediction),
        pos_p_val=pos_p_val.item(),
        p_value_threshold=threshold,
    )


# todo - this belongs in flomo repo
# def handle_getting_input_dir(dir_path, local_dir):
#     dir_name = os.path.basename(dir_path.rstrip("/"))
#     new_local_dir = os.path.join(local_dir, dir_name)
#     os.makedirs(new_local_dir, exist_ok=True)
#
#     if os.path.exists(dir_path):
#         return dir_path
#     if "s3://" in dir_path:
#         for file in list_files_in_s3_directory(dir_path):
#             download_file_from_s3(file, new_local_dir)
#     return new_local_dir
#
#
# def handle_getting_input_file(input_file_path, local_dir):
#     """
#
#     :param input_file_path: The s3 path or the local file path.
#     :param local_dir: If s3 file, we will download to this local_dir
#     :return:
#     """
#     if os.path.exists(input_file_path):
#         return input_file_path
#     if "s3://" in input_file_path:
#         return download_file_from_s3(input_file_path, local_dir)
#     raise RuntimeError(f"Couldnt find {input_file_path=}")


@dataclass
class PredictionModel:
    model_file_path: str
    model_type: str

    def __post_init__(self):
        try:
            model_items = pickle.load(open(self.model_file_path, "rb"))
            self.model = model_items["model"]
            self.feat_indexes = model_items["feat_indexes"] if "feat_indexes" in model_items else None
            self.som_dim = model_items["som_dim"]
            self.tube_names = model_items["tube_names"]
            self.diagnoses = model_items["diagnoses"]
            self.p_value_threshold = model_items["p_value_threshold"]
            self.scale_range = model_items["scale_range"] if "scale_range" in model_items else None
            self.normalization_sample_count = (
                model_items["normalization_sample_count"] if "normalization_sample_count" in model_items else None
            )
            self.filter_data = model_items["filter_data"] if "filter_data" in model_items else None

            # todo - ok to drop model-specific version, right?
            # model_version_attr = f"MODEL_VERSION_{self.model_type.upper()}"
            # self.model_version = getattr(config, model_version_attr)

            # Save the md5sum of the whole model pickle, just as extra insurance we know what model we are working with.
            with open(self.model_file_path, "rb") as file:
                file_data = file.read()
                self.md5sum = hashlib.md5(file_data).hexdigest()
        except Exception:
            logger.error("Failed to load model.")
            raise


async def process_case(
    # jobid,
    accession,
    local_b_tube,
    local_t_tube,
    local_mye_tube,
    b_cell_cytometer_id,
    t_cell_cytometer_id,
    mye_cytometer_id,
    b_cell_run_date,
    t_cell_run_date,
    mye_run_date,
    local_model_path,
    centroid_paths,
    predict_model_type,
    prediction_category,
    tube_metrics,
    local_dir="/tmp",
):
    logger.info(
        f"Starting process_case of {prediction_category=} function with {predict_model_type=} " f"for {accession=}"
    )
    # Create a unique tmp dir for this run based on jobid.
    # local_dir = os.path.join(local_dir, jobid)
    os.makedirs(local_dir, exist_ok=True)

    # todo - need to put this somewhere else in flomo or conditionally call it
    # stream: FlomoAccessionStream = await FlomoAccessionStream.initialize(accession)

    # todo - need to put this somewhere else in flomo or conditionally call it
    # await stream.emit(FlomoTaskStarted, FlomoTaskStartedData(accession=accession, jobid=jobid))

    # Download tube files, centroid files, and model pickle file.
    output_features = os.path.join(local_dir, "output_features.npy")

    # todo - belongs in flomo
    # local_b_tube = handle_getting_input_file(b_cell_tube, local_dir)
    # local_t_tube = handle_getting_input_file(t_cell_tube, local_dir)
    # local_mye_tube = handle_getting_input_file(mye_tube, local_dir)
    # local_model_path = handle_getting_input_file(predict_model_path, local_dir)
    # local_centroid_dir = handle_getting_input_dir(centroid_dir, local_dir)

    model = PredictionModel(local_model_path, prediction_category)

    logger.info(
        f"Prediction model: {model.diagnoses=}, {model.p_value_threshold=}, {model.som_dim=}, "
        f"{model.tube_names=}, {model=}, {model.feat_indexes=}, {model.md5sum=}, "
        f"{model.scale_range=} {model.filter_data=}"
    )

    b_tube_df, t_tube_df, mye_tube_df = (
        pd.read_csv(local_b_tube),
        pd.read_csv(local_t_tube),
        pd.read_csv(local_mye_tube),
    )

    if model.filter_data:
        logger.info("Preparing to filter tube data")
        # data from old software version was already prefiltered
        # and passes through this step unaltered
        (b_tube_df, t_tube_df, mye_tube_df) = filter_multi(b_tube_df, t_tube_df, mye_tube_df)

    # data from old software version v1.0 will not have "event_filter_keep" column
    # and is NOT compatible with unfiltered analysis (normal, viability)
    b_t_m_dfs = []
    prefiltered = []
    for df in (b_tube_df, t_tube_df, mye_tube_df):
        if "event_filter_keep" in df.columns:
            b_t_m_dfs.append(df.drop(columns=["event_filter_keep", "Time"], axis=1))
            prefiltered.append(False)
        else:
            b_t_m_dfs.append(df.drop(columns=["Time"], axis=1))
            prefiltered.append(True)
    b_tube_df, t_tube_df, mye_tube_df = b_t_m_dfs

    # can only predict for unfiltered models if the data is not prefiltered
    filtering_ok = False if (any(prefiltered) and not model.filter_data) else True

    tube_dfs = await normalize1case(
        accession=accession,
        b_cell_tube=b_tube_df,
        t_cell_tube=t_tube_df,
        mye_tube=mye_tube_df,
        b_cell_cytometer_id=b_cell_cytometer_id,
        t_cell_cytometer_id=t_cell_cytometer_id,
        mye_cytometer_id=mye_cytometer_id,
        b_cell_run_date=b_cell_run_date,
        t_cell_run_date=t_cell_run_date,
        mye_run_date=mye_run_date,
        normalization_sample_count=model.normalization_sample_count,
        df_metrics=pd.DataFrame(tube_metrics),
    )

    features = project1case(
        input_tubes=tube_dfs,
        # centroid_dir=local_centroid_dir,
        centroid_paths=centroid_paths,
        output_features=output_features,
        diagnoses=model.diagnoses,
        tube_names=model.tube_names,
        som_dim=model.som_dim,
        num_workers=1,
        normalize_soms=True,
    )

    if filtering_ok:
        res = predict1case(
            features=features,
            model=model.model,
            feat_indexes=model.feat_indexes,
            som_dim=model.som_dim,
            tube_names=model.tube_names,
            diagnoses=model.diagnoses,
            threshold=model.p_value_threshold,
            scale_range=model.scale_range,
        )

        prediction = Prediction(
            disease=prediction_category,
            prediction=res["prediction"],
            pValue=res["pos_p_val"],
            pValueThreshold=res["p_value_threshold"],
            metadata=PredictionMetadata(
                modelVersion="NA",
                modelMD5Sum=model.md5sum,
                classifierLocation=local_model_path,
                centroidLocation=", ".join(centroid_paths),
            ),
        )
    else:
        logger.warning(
            "At least one tube file was prefiltered (i.e., had no 'event_filter_keep' column). "
            "This model is not compatible with filtered data."
        )
        logger.warning(f"Prefiltered status for (b, t, m) tubes is {prefiltered}")

    return prediction
