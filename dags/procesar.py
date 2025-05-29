from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
import os


MLFLOW_TRACKING_URI = "http://10.43.101.175:30500"
MLFLOW_S3_ENDPOINT_URL = "http://10.43.101.175:30382"
AWS_ACCESS_KEY_ID = "adminuser"
AWS_SECRET_ACCESS_KEY = "securepassword123"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    '2-Procesar_data',
    default_args=default_args,
    description='DAG para procesar datos raw y guardar en cleandata con tracking MLflow',
    schedule_interval='3 0 * * *',
    start_date=datetime(2025, 5, 28),
    catchup=False,
    max_active_runs=1
)

raw_schema = 'rawdata'
raw_table = 'houses'
clean_schema = 'cleandata'
clean_table = 'processed_houses'

def set_mlflow_tracking(**kwargs):
    """Configurar tracking de MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

    print("✅ Tracking de MLflow configurado exitosamente")

create_schema_and_table_sql = f"""
CREATE SCHEMA IF NOT EXISTS {clean_schema};

CREATE TABLE IF NOT EXISTS {clean_schema}.{clean_table} (
    id SERIAL PRIMARY KEY,
    brokered_by VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    price NUMERIC(12,2) NOT NULL,
    bed INT NOT NULL,
    bath NUMERIC(3,1) NOT NULL,
    acre_lot NUMERIC(8,3) NOT NULL,
    street VARCHAR(150) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50) NOT NULL,
    zip_code VARCHAR(20) NOT NULL,
    house_size INT NOT NULL,
    prev_sold_date DATE,
    price_per_sqft NUMERIC(12,4)
);
"""

def process_data(**kwargs):
    
    mlflow.set_experiment('house_price_pipeline')
    
    with mlflow.start_run(run_name="data_processing"):
        try:
            
            hook = PostgresHook(postgres_conn_id='postgres_default')
            engine = hook.get_sqlalchemy_engine()

            
            query = f"SELECT * FROM {raw_schema}.{raw_table};"
            df = pd.read_sql(query, con=engine)
            
            
            initial_count = len(df)
            mlflow.log_metric("raw_records_input", initial_count)
            mlflow.log_metric("initial_null_values", df.isnull().sum().sum())

            
            
            df_clean = df[(df['price'] > 0) & (df['house_size'] > 0)]
            
            
            filtered_count = len(df_clean)
            mlflow.log_metric("records_after_price_filter", filtered_count)
            mlflow.log_metric("records_removed", initial_count - filtered_count)

            
            df_clean['price_per_sqft'] = df_clean['price'] / df_clean['house_size']

            
            df_clean['status'] = df_clean['status'].str.lower()

            
            df_clean['prev_sold_date'] = pd.to_datetime(df_clean['prev_sold_date'], errors='coerce')

            
            mlflow.log_metric("final_records_count", len(df_clean))
            mlflow.log_metric("final_null_values", df_clean.isnull().sum().sum())
            
            
            mlflow.log_metric("price_mean_processed", df_clean['price'].mean())
            mlflow.log_metric("price_median_processed", df_clean['price'].median())
            mlflow.log_metric("price_std_processed", df_clean['price'].std())
            mlflow.log_metric("price_per_sqft_mean", df_clean['price_per_sqft'].mean())
            mlflow.log_metric("house_size_mean", df_clean['house_size'].mean())
            mlflow.log_metric("unique_cities_processed", df_clean['city'].nunique())
            mlflow.log_metric("unique_states_processed", df_clean['state'].nunique())

            
            mlflow.log_metric("bed_mean", df_clean['bed'].mean())
            mlflow.log_metric("bath_mean", df_clean['bath'].mean())
            mlflow.log_metric("acre_lot_mean", df_clean['acre_lot'].mean())

            
            mlflow.log_param("filter_applied", "price > 0 AND house_size > 0")
            mlflow.log_param("new_feature_created", "price_per_sqft")
            mlflow.log_param("status_transformation", "lowercase")
            
            
            df_clean.to_sql(
                name=clean_table,
                con=engine,
                schema=clean_schema,
                if_exists='replace',
                index=False,
                chunksize=1000,
                method='multi'
            )

            
            mlflow.log_param("processing_status", "success")
            print(f"Filas procesadas y guardadas: {len(df_clean)}")

        except Exception as e:
            mlflow.log_param("processing_status", "error")
            mlflow.log_param("error_message", str(e))
            print(f"Error en procesamiento: {e}")
            raise


set_mlflow_tracking_task = PythonOperator(
    task_id='set_mlflow_tracking',
    python_callable=set_mlflow_tracking,
    dag=dag
)

create_table_task = PostgresOperator(
    task_id='create_schema_and_table_processed',
    postgres_conn_id='postgres_default',
    sql=create_schema_and_table_sql,
    dag=dag
)

process_data_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag
)


set_mlflow_tracking_task >> create_table_task >> process_data_task