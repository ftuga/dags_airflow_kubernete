from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import joblib
import os

# Evidently API nueva
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 27),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'email_on_retry': False
}

dag = DAG(
    '3-Entrenar_modelo_con_drift_evidently',
    default_args=default_args,
    schedule_interval='5 0 * * *',
    catchup=False,
    max_active_runs=1,
    description='DAG para entrenar modelo con chequeo de data drift usando Evidently'
)

clean_schema = 'cleandata'
clean_table = 'processed_houses'
previous_data_path = '/tmp/previous_training_data.joblib'

def check_data_count(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    sql = f"SELECT COUNT(*) FROM {clean_schema}.{clean_table};"
    records = hook.get_first(sql)
    count = records[0] if records else 0
    print(f"Registros en tabla {clean_schema}.{clean_table}: {count}")
    if count > 20000:
        return "detect_data_drift_task"
    else:
        return "skip_training_task"

def detect_data_drift(**kwargs):
    import os
    import joblib
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    query = f"SELECT * FROM {clean_schema}.{clean_table};"
    df_new = pd.read_sql(query, con=engine)

    # Mantener solo columnas relevantes
    relevant_cols = ['price', 'bed', 'bath', 'acre_lot']
    df_new = df_new[relevant_cols]

    # Convertir object a category
    categorical_cols = df_new.select_dtypes(include='object').columns.tolist()
    df_new[categorical_cols] = df_new[categorical_cols].astype('category')

    numerical_cols = [col for col in df_new.columns if col not in categorical_cols + ['price']]

    column_mapping = ColumnMapping(
        target='price',
        numerical_features=numerical_cols,
        categorical_features=categorical_cols
    )

    if not os.path.exists(previous_data_path):
        joblib.dump(df_new, previous_data_path)
        print("No hay datos anteriores, se asume no drift.")
        return "train_model_task"

    df_old = joblib.load(previous_data_path)

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=df_old, current_data=df_new, column_mapping=column_mapping)
    result = report.as_dict()

    drift_flag = result['metrics'][0]['result'].get('dataset_drift', False)
    target_drift_score = result['metrics'][1]['result'].get('drift_score', 0.0)

    if drift_flag or target_drift_score > 0.05:
        print(f"Drift detectado: dataset_drift={drift_flag}, target_drift_score={target_drift_score}")
        return "skip_training_task"
    else:
        joblib.dump(df_new, previous_data_path)
        print("No hay drift detectado, se continúa al entrenamiento.")
        return "train_model_task"


def train_model(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    query = f"SELECT * FROM {clean_schema}.{clean_table};"
    df = pd.read_sql(query, con=engine)

    y = df['price']
    X = df.drop(columns=['id', 'price', 'prev_sold_date'])
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Guardar subconjunto para comparación futura
    df_train = X_train.copy()
    df_train['price'] = y_train
    joblib.dump(df_train, previous_data_path)
    print(f"Datos de entrenamiento guardados en {previous_data_path}")

    model = LGBMRegressor(
        objective='regression',
        n_estimators=50,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)]
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"RMSE validación: {rmse:.4f}")

    model.booster_.save_model('/tmp/lgbm_model.txt')
    print("Modelo guardado en /tmp/lgbm_model.txt")

with dag:

    check_data_task = BranchPythonOperator(
        task_id='check_data_count',
        python_callable=check_data_count
    )

    detect_data_drift_task = BranchPythonOperator(
        task_id='detect_data_drift_task',
        python_callable=detect_data_drift
    )

    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model
    )

    skip_training_task = DummyOperator(
        task_id='skip_training_task'
    )

    end_task = DummyOperator(
        task_id='end_task',
        trigger_rule='none_failed_min_one_success'
    )

    check_data_task >> detect_data_drift_task >> [train_model_task, skip_training_task] >> end_task
