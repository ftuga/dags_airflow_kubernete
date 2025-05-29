from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import BranchPythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
import pandas as pd
import requests
import json
import time
import os
import mlflow


MLFLOW_TRACKING_URI = "http://10.43.101.175:30500"
MLFLOW_S3_ENDPOINT_URL = "http://10.43.101.175:30382"
AWS_ACCESS_KEY_ID = "adminuser"
AWS_SECRET_ACCESS_KEY = "securepassword123"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0
}

dag = DAG(
    '1-Cargar_data',
    default_args=default_args,
    description='DAG para cargar datos desde el servidor a PostgreSQL con tracking MLflow',
    schedule_interval='1 0 * * *',
    start_date=datetime(2025, 5, 27, 0, 0, 0),
    catchup=False,
    max_active_runs=1
)

database_name = 'rawdata'
table_name = 'houses'

def set_mlflow_tracking(**kwargs):
    """Configurar tracking de MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

    print("✅ Tracking de MLflow configurado exitosamente")

create_schema_and_table_sql = f"""
CREATE SCHEMA IF NOT EXISTS {database_name};

CREATE TABLE IF NOT EXISTS {database_name}.{table_name} (
    id SERIAL PRIMARY KEY,
    brokered_by VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    price NUMERIC(12,2) NOT NULL,
    bed INT NOT NULL,
    bath NUMERIC(8,3) NOT NULL,
    acre_lot NUMERIC(12,3) NOT NULL,
    street VARCHAR(150) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50) NOT NULL,
    zip_code VARCHAR(20) NOT NULL,
    house_size INT NOT NULL,
    prev_sold_date DATE
);
"""

def server_response(group_number=1, max_retries=3, wait_seconds=5):
    server_url = 'http://10.43.101.108:80/data'
    server_url_restart = 'http://10.43.101.108:80/restart_data_generation'
    params = {"group_number": group_number, "day": "Tuesday"}

    retries = 0
    while retries < max_retries:
        response = requests.get(server_url, params=params)

        if response.status_code == 200:
            return response

        elif response.status_code == 400:
            try:
                detail = response.json().get('detail', '')
            except Exception:
                detail = ''

            if detail == "Ya se recolectó toda la información mínima necesaria":
                response_restart = requests.get(server_url_restart, params=params)
                if response_restart.status_code == 200:
                    time.sleep(wait_seconds)
                    retries += 1
                    continue
                else:
                    return response_restart
            else:
                return response
        else:
            return response

    raise Exception("No se pudo obtener datos válidos luego de reiniciar la generación")

def load_data(**kwargs):
    
    mlflow.set_experiment('house_price_pipeline')
    
    with mlflow.start_run(run_name="data_ingestion"):
        try:
            raw = server_response()
            data = json.loads(raw.content.decode('utf-8'))

            
            mlflow.log_param("data_source", "external_api")
            mlflow.log_param("group_number", 1)
            mlflow.log_param("day", "Tuesday")

            
            df = pd.DataFrame(data["data"], columns=[
                "brokered_by", "status", "price", "bed", "bath",
                "acre_lot", "street", "city", "state", "zip_code",
                "house_size", "prev_sold_date"
            ])

            
            mlflow.log_metric("raw_records_received", len(df))

            
            df["price"] = pd.to_numeric(df["price"], errors='coerce')
            df["bed"] = pd.to_numeric(df["bed"], errors='coerce').fillna(0).astype(int)
            df["bath"] = pd.to_numeric(df["bath"], errors='coerce')
            df["acre_lot"] = pd.to_numeric(df["acre_lot"], errors='coerce')
            df["house_size"] = pd.to_numeric(df["house_size"], errors='coerce').fillna(0).astype(int)
            df["prev_sold_date"] = pd.to_datetime(df["prev_sold_date"], errors='coerce')

            
            null_counts = df.isnull().sum()
            mlflow.log_metric("null_values_total", null_counts.sum())
            mlflow.log_metric("price_mean", df["price"].mean())
            mlflow.log_metric("price_std", df["price"].std())
            mlflow.log_metric("unique_cities", df["city"].nunique())

            
            postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
            engine = postgres_hook.get_sqlalchemy_engine()

            
            create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {database_name};"
            postgres_hook.run(create_schema_sql)

            
            df.to_sql(
                name=table_name,
                con=engine,
                schema=database_name,
                if_exists='append',
                index=False,
                chunksize=1000,
                method='multi'
            )

            
            count_query = f"SELECT COUNT(*) FROM {database_name}.{table_name}"
            records_count = postgres_hook.get_records(count_query)[0][0]
            
            mlflow.log_metric("total_records_in_db", records_count)
            mlflow.log_metric("records_loaded_this_run", len(df))
            
            print(f"Filas cargadas: {records_count}")
            
            
            mlflow.log_param("status", "success")

        except Exception as e:
            mlflow.log_param("status", "error")
            mlflow.log_param("error_message", str(e))
            print(f"Error en carga de datos: {e}")
            raise

    time.sleep(2)

def decide_next_task(**kwargs):
    iter_count = Variable.get("dag_iter_count", default_var=1)
    iter_count = int(iter_count)
    time.sleep(5)
    if iter_count > 10:
        return "stop_task"
    else:
        return "load_data"


set_mlflow_tracking_task = PythonOperator(
    task_id='set_mlflow_tracking',
    python_callable=set_mlflow_tracking,
    dag=dag
)

create_table_task = PostgresOperator(
    task_id='create_schema_and_table',
    postgres_conn_id='postgres_default',
    sql=create_schema_and_table_sql,
    dag=dag
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag
)


set_mlflow_tracking_task >> create_table_task >> load_data_task