from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd

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
    description='DAG para procesar datos raw y guardar en cleandata',
    schedule_interval='3 0 * * *',
    start_date=datetime(2025, 5, 28),
    catchup=False,
    max_active_runs=1
)

raw_schema = 'rawdata'
raw_table = 'houses'

clean_schema = 'cleandata'
clean_table = 'processed_houses'

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
    price_per_sqft NUMERIC(12,4)  -- ejemplo de variable calculada
);
"""

def process_data(**kwargs):
    # Conexión a Postgres
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()

    # Leer datos raw
    query = f"SELECT * FROM {raw_schema}.{raw_table};"
    df = pd.read_sql(query, con=engine)

    # Ejemplo de limpieza y transformación
    # Eliminar filas con price o house_size nulos o cero
    df = df[(df['price'] > 0) & (df['house_size'] > 0)]

    # Crear variable nueva: precio por pie cuadrado
    df['price_per_sqft'] = df['price'] / df['house_size']

    # Convertir status a minúsculas (ejemplo)
    df['status'] = df['status'].str.lower()

    # Rellenar valores nulos en prev_sold_date con None (ya debería estar como datetime)
    df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')

    # Guardar datos procesados en cleandata.processed_houses
    df.to_sql(
        name=clean_table,
        con=engine,
        schema=clean_schema,
        if_exists='replace',  # reemplaza siempre con los datos procesados más recientes
        index=False,
        chunksize=1000,
        method='multi'
    )

    # Log de filas procesadas
    print(f"Filas procesadas y guardadas: {len(df)}")


# Crear esquema y tabla antes de insertar
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

create_table_task >> process_data_task
