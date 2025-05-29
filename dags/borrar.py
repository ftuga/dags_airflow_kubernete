from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 28),
    'retries': 0,
    'email_on_failure': False,
    'email_on_retry': False
}

dag = DAG(
    '0-Borrar_esquemas',
    default_args=default_args,
    schedule_interval='0 0 * * *',
    catchup=False,
    max_active_runs=1,
    description='DAG para borrar esquemas rawdata y cleandata antes de otros DAGs'
)

drop_schemas_sql = """
DROP SCHEMA IF EXISTS rawdata CASCADE;  
DROP SCHEMA IF EXISTS cleandata CASCADE;
"""
def check_schemas(**kwargs):
    hook = PostgresHook(postgres_conn_id='postgres_default')
    engine = hook.get_sqlalchemy_engine()
    with engine.connect() as conn:
        result = conn.execute("SELECT schema_name FROM information_schema.schemata;")
        schemas = [row[0] for row in result]
        print("Schemas disponibles:", schemas)

check_schemas_task = PythonOperator(
    task_id='check_schemas',
    python_callable=check_schemas,
    dag=dag
)


drop_schemas_task = PostgresOperator(
    task_id='drop_schemas',
    postgres_conn_id='postgres_default',
    sql=drop_schemas_sql,
    dag=dag
)
check_schemas_task >> drop_schemas_task
