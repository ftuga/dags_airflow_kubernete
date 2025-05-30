from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import mlflow
import os


MLFLOW_TRACKING_URI = "http://10.43.101.175:30500"
MLFLOW_S3_ENDPOINT_URL = "http://10.43.101.175:30382"
AWS_ACCESS_KEY_ID = "adminuser"
AWS_SECRET_ACCESS_KEY = "securepassword123"

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

def set_mlflow_tracking(**kwargs):
    """Configurar tracking de MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY
    print("✅ Tracking de MLflow configurado exitosamente")

def check_schemas(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    result = pg_hook.get_records("SELECT schema_name FROM information_schema.schemata;")
    schemas = [row[0] for row in result]
    print("Schemas disponibles:", schemas)

def drop_schemas(**kwargs):
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    pg_hook.run("""
        DROP SCHEMA IF EXISTS rawdata CASCADE;  
        DROP SCHEMA IF EXISTS cleandata CASCADE;
    """)
    print("✅ Schemas rawdata y cleandata eliminados")

def log_cleanup_mlflow(**kwargs):
    """Log de limpieza en MLflow"""
    try:
        mlflow.set_experiment('house_price_pipeline')
        with mlflow.start_run(run_name="schema_cleanup"):
            mlflow.log_param("operation", "schema_cleanup")
            mlflow.log_param("schemas_dropped", "rawdata, cleandata")
            mlflow.log_metric("cleanup_timestamp", datetime.now().timestamp())
            print("✅ Cleanup registrado en MLflow")
    except Exception as e:
        print(f"❌ Error al registrar en MLflow: {e}")


set_mlflow_tracking_task = PythonOperator(
    task_id='set_mlflow_tracking',
    python_callable=set_mlflow_tracking,
    dag=dag
)

check_schemas_task = PythonOperator(
    task_id='check_schemas',
    python_callable=check_schemas,
    dag=dag
)

drop_schemas_task = PythonOperator(
    task_id='drop_schemas',
    python_callable=drop_schemas,
    dag=dag
)

log_cleanup_task = PythonOperator(
    task_id='log_cleanup_mlflow',
    python_callable=log_cleanup_mlflow,
    dag=dag
)


set_mlflow_tracking_task >> check_schemas_task >> drop_schemas_task >> log_cleanup_task