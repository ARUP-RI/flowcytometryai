from flowcytometryai.predict import process_case as predict_pc
from flowcytometryai.enums import Disease, ModelType


async def predict(
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
    b_centroids_path,
    t_centroids_path,
    m_centroids_path,
    tube_metrics,
    local_dir="/tmp/aml",
):
    prediction_type = Disease.AML
    predict_model_type = ModelType.BOOST
    centroid_paths = [b_centroids_path, t_centroids_path, m_centroids_path]

    prediction = await predict_pc(
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
        prediction_type,
        local_dir=local_dir,
    )

    return prediction
