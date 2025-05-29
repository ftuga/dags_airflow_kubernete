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
    description='DAG para cargar datos desde el servidor a PostgreSQL sin preprocesamiento',
    schedule_interval='1 0 * * *',  # Solo ejecución manual si es "None"
    start_date=datetime(2025, 5, 27, 0, 0, 0),
    catchup=False,
    max_active_runs=1
)

database_name = 'rawdata'  # Name of the database (used as schema if needed)
table_name = 'houses'

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
            # Si la respuesta es exitosa, la retornamos
            return response

        elif response.status_code == 400:
            try:
                detail = response.json().get('detail', '')
            except Exception:
                detail = ''

            if detail == "Ya se recolectó toda la información mínima necesaria":
                # Hacemos restart
                response_restart = requests.get(server_url_restart, params=params)
                if response_restart.status_code == 200:
                    # Esperamos un poco para que la generación reinicie
                    time.sleep(wait_seconds)
                    retries += 1
                    continue  # Intentamos obtener datos de nuevo
                else:
                    # Si no pudo reiniciar, retornamos error
                    return response_restart
            else:
                # Error 400 pero con otro detalle, retornamos el error
                return response
        else:
            # Otro error diferente, lo retornamos
            return response

    # Si llegamos aquí, es porque se superó el máximo número de reintentos
    raise Exception("No se pudo obtener datos válidos luego de reiniciar la generación")


def load_data(**kwargs):
    raw = server_response()  # Se asume que esta función está definida y maneja la lógica HTTP

    data = json.loads(raw.content.decode('utf-8'))

    # Ajusta las columnas según la nueva tabla
    df = pd.DataFrame(data["data"], columns=[
        "brokered_by", "status", "price", "bed", "bath",
        "acre_lot", "street", "city", "state", "zip_code",
        "house_size", "prev_sold_date"
    ])

    # Tipo de datos
    df["price"] = pd.to_numeric(df["price"], errors='coerce')
    df["bed"] = pd.to_numeric(df["bed"], errors='coerce').fillna(0).astype(int)
    df["bath"] = pd.to_numeric(df["bath"], errors='coerce')
    df["acre_lot"] = pd.to_numeric(df["acre_lot"], errors='coerce')
    df["house_size"] = pd.to_numeric(df["house_size"], errors='coerce').fillna(0).astype(int)
    df["prev_sold_date"] = pd.to_datetime(df["prev_sold_date"], errors='coerce')

    # Conexión a Postgres
    postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = postgres_hook.get_sqlalchemy_engine()

    # # Crear schema si no existe
    create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {database_name};"
    postgres_hook.run(create_schema_sql)

    # Insertar datos en tabla
    df.to_sql(
        name=table_name,
        con=engine,
        schema=database_name,
        if_exists='append',  # NO 'replace' ni 'fail'
        index=False,
        chunksize=1000,
        method='multi'  # opcional, para eficiencia
    )


    # Contar registros cargado
    count_query = f"SELECT COUNT(*) FROM {database_name}.{table_name}"
    records_count = postgres_hook.get_records(count_query)[0][0]
    print(f"Filas cargadas: {records_count}")

    time.sleep(2)


def decide_next_task(**kwargs):
    iter_count = Variable.get("dag_iter_count", default_var=1)
    iter_count = int(iter_count)
    time.sleep(5)
    if iter_count > 10:
        return "stop_task"
    else:
        return "load_data"


# PostgreSQL Operator to create the table (with the correct schema handling)
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

create_table_task >> load_data_task