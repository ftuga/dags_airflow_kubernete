from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
import boto3


MLFLOW_TRACKING_URI = "http://10.43.101.175:30500"
MLFLOW_S3_ENDPOINT_URL = "http://10.43.101.175:30382"
AWS_ACCESS_KEY_ID = "adminuser"
AWS_SECRET_ACCESS_KEY = "securepassword123"
bucket_name = "mlflow-artifacts"


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
    description='DAG para entrenar modelo con chequeo de data drift usando Evidently y MLflow'
)

clean_schema = 'cleandata'
clean_table = 'processed_houses'
previous_data_path = '/tmp/previous_training_data.joblib'

def set_mlflow_tracking(**kwargs):
    """Configurar tracking de MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

    print("✅ Tracking de MLflow configurado exitosamente")

def check_data_count(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    sql = f"SELECT COUNT(*) FROM {clean_schema}.{clean_table};"
    records = hook.get_first(sql)
    count = records[0] if records else 0
    print(f"Registros en tabla {clean_schema}.{clean_table}: {count}")
    
    
    mlflow.set_experiment('house_price_pipeline')
    with mlflow.start_run(run_name="data_count_check"):
        mlflow.log_metric("data_count", count)
        mlflow.log_param("threshold", 20000)
    
    if count > 20000:
        return "detect_data_drift_task"
    else:
        return "skip_training_task"

def detect_data_drift(**kwargs):
    mlflow.set_experiment('house_price_pipeline')
    
    with mlflow.start_run(run_name="drift_detection"):
        hook = PostgresHook(postgres_conn_id='postgres_default')
        engine = hook.get_sqlalchemy_engine()
        query = f"SELECT * FROM {clean_schema}.{clean_table};"
        df_new = pd.read_sql(query, con=engine)

        
        relevant_cols = ['price', 'bed', 'bath', 'acre_lot']
        df_new = df_new[relevant_cols]

        
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
            mlflow.log_param("drift_status", "no_reference_data")
            print("No hay datos anteriores, se asume no drift.")
            return "train_model_task"

        df_old = joblib.load(previous_data_path)

        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=df_old, current_data=df_new, column_mapping=column_mapping)
        result = report.as_dict()

        drift_flag = result['metrics'][0]['result'].get('dataset_drift', False)
        target_drift_score = result['metrics'][1]['result'].get('drift_score', 0.0)

        
        mlflow.log_metric("dataset_drift_detected", int(drift_flag))
        mlflow.log_metric("target_drift_score", target_drift_score)
        mlflow.log_param("drift_threshold", 0.05)

        if drift_flag or target_drift_score > 0.05:
            mlflow.log_param("drift_decision", "skip_training")
            print(f"Drift detectado: dataset_drift={drift_flag}, target_drift_score={target_drift_score}")
            return "skip_training_task"
        else:
            joblib.dump(df_new, previous_data_path)
            mlflow.log_param("drift_decision", "continue_training")
            print("No hay drift detectado, se continúa al entrenamiento.")
            return "train_model_task"

def train_model(**kwargs):
    mlflow.set_experiment('house_price_pipeline')
    
    with mlflow.start_run(run_name="model_training_comparison") as main_run:
        hook = PostgresHook(postgres_conn_id='postgres_default')
        engine = hook.get_sqlalchemy_engine()
        query = f"SELECT * FROM {clean_schema}.{clean_table};"
        df = pd.read_sql(query, con=engine)

        y = df['price']
        X = df.drop(columns=['id', 'price', 'prev_sold_date'])
        for col in X.select_dtypes(include='object').columns:
            X[col] = X[col].astype('category')

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        
        df_train = X_train.copy()
        df_train['price'] = y_train
        joblib.dump(df_train, previous_data_path)
        print(f"Datos de entrenamiento guardados en {previous_data_path}")

        
        models = {
            'LightGBM': LGBMRegressor(
                objective='regression',
                n_estimators=100,
                random_state=42,
                n_jobs=1
            ),
            'RandomForest': RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=1
            ),
            'LinearRegression': LinearRegression()
        }

        def evaluate_model(y_true, y_pred):
            return {
                'rmse': mean_squared_error(y_true, y_pred, squared=False),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }

        best_model = None
        best_score = float('inf')
        best_model_obj = None
        model_metrics = {}
        client = mlflow.tracking.MlflowClient()

        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("val_size", len(X_val))
        mlflow.log_param("features", list(X.columns))

        for name, model in models.items():
            print(f"Entrenando modelo: {name}")
            with mlflow.start_run(nested=True):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                metrics = evaluate_model(y_val, y_pred)
                
                
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                mlflow.log_param('model_name', name)
                
                
                try:
                    mlflow.sklearn.log_model(
                        model, 
                        name,
                        registered_model_name=f"house_price_{name}"
                    )
                    print(f"Modelo {name} registrado exitosamente")
                except Exception as e:
                    print(f"Error al registrar el modelo {name}: {e}")
                
                model_metrics[name] = metrics['rmse']
                
                if metrics['rmse'] < best_score:
                    best_score = metrics['rmse']
                    best_model = name
                    best_model_obj = model

                print(f"Modelo {name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2_score']:.4f}")

        
        mlflow.log_param("best_model_name", best_model)
        mlflow.log_metric("best_rmse", best_score)

        
        if best_model:
            print(f"Mejor modelo: {best_model} con RMSE: {best_score}")
            
            try:
                model_name = f"house_price_{best_model}"
                versions = client.search_model_versions(f"name='{model_name}'")
                
                if versions and len(versions) > 0:
                    latest_version = max([int(v.version) for v in versions])
                    
                    
                    for version in versions:
                        if version.current_stage == "Production":
                            client.transition_model_version_stage(
                                name=model_name,
                                version=version.version,
                                stage="Archived"
                            )
                    
                    
                    client.transition_model_version_stage(
                        name=model_name, 
                        version=latest_version, 
                        stage="Production"
                    )
                    
                    print(f"Modelo {best_model} versión {latest_version} promovido a Producción")
                    mlflow.log_param("production_model_version", latest_version)
                    
                else:
                    print(f"No se encontraron versiones para el modelo {model_name}")
                    
            except Exception as e:
                print(f"Error al promover modelo a producción: {e}")
                mlflow.log_param("production_promotion_error", str(e))

        
        if best_model_obj:
            joblib.dump(best_model_obj, '/tmp/best_house_price_model.joblib')
            print("Mejor modelo guardado localmente")

with dag:
    set_mlflow_tracking_task = PythonOperator(
        task_id='set_mlflow_tracking',
        python_callable=set_mlflow_tracking
    )

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

    set_mlflow_tracking_task >> check_data_task >> detect_data_drift_task >> [train_model_task, skip_training_task] >> end_task