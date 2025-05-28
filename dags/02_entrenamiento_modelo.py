import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.dummy import DummyOperator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import boto3

# Configuración de variables de entorno
MLFLOW_TRACKING_URI = "http://10.43.101.175:30500"
MLFLOW_S3_ENDPOINT_URL = "http://10.43.101.175:30382"
AWS_ACCESS_KEY_ID = "adminuser"
AWS_SECRET_ACCESS_KEY = "securepassword123"
HOST_IP = "10.43.101.175"
bucket_name = "mlflow-artifacts"
object_key = "preprocessors/preprocessor.joblib"  # Path within the bucket

# Límites de recursos para evitar sobrecargar la máquina
MAX_THREADS = 1  # Usar un solo hilo para LightGBM
SAMPLE_SIZE = 0.1  # Usar solo 10% de la muestra para entrenar
DATA_SAMPLE_SIZE = 0.1  # Extraer solo 10% de los datos

def set_mlflow_tracking(**kwargs):
    """Configurar tracking de MLflow"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment('diabetes_experiment')
    
    # Configurar credenciales de almacenamiento
    os.environ['MLFLOW_TRACKING_URI'] = MLFLOW_TRACKING_URI
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY_ID
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_ACCESS_KEY

    print("✅ Tracking de MLflow configurado exitosamente")


def load_data(**kwargs):
    """Cargar datos desde PostgreSQL para entrenamiento de modelos con muestreo"""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    # Query extremadamente simplificada para minimizar datos
    query = f"""
    SELECT * FROM clean_data.diabetes_train 
    WHERE batch_id = (
        SELECT MIN(batch_id) 
        FROM clean_data.batch_info
        WHERE batch_id IS NOT NULL
    )
    LIMIT 500  -- Usar un valor fijo pequeño en lugar de un porcentaje
    """
        
    df = pg_hook.get_pandas_df(query)
    
    if df.empty:
        print("No hay más datos para procesar")
        return {'continue_processing': False}
    
    # Hacer un muestreo adicional para reducir la carga
    if len(df) > 1000:  # Si hay muchos registros
        df = df.sample(frac=SAMPLE_SIZE, random_state=42)
    
    columns_to_drop = ['id', 'batch_id', 'dataset']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Guardar a un archivo temporal en lugar de usar XCom para grandes conjuntos de datos
    tmp_file = '/tmp/diabetes_train_data.csv'
    df.to_csv(tmp_file, index=False)
    
    # Solo almacenar la ruta del archivo en XCom, no los datos completos
    kwargs['ti'].xcom_push(key='training_data_path', value=tmp_file)
    
    return {
        'training_data_path': tmp_file,
        'continue_processing': True
    }


def preprocess_data(**kwargs):
    """Preprocesar datos de entrenamiento con eficiencia de memoria mejorada"""
    training_data_path = kwargs['ti'].xcom_pull(key='training_data_path')
    
    # Cargar desde archivo temporal
    df = pd.read_csv(training_data_path)
    
    y_train = df['readmitted'].tolist()
    X_train = df.drop('readmitted', axis=1)
    
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Realizar fit_transform una sola vez
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Guardar el preprocesador
    local_path = '/tmp/preprocessor.joblib'
    with open(local_path, 'wb') as f:
        joblib.dump(preprocessor, f)

    # Subir a MinIO
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=MLFLOW_S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print("✅ MinIO client initialized successfully.")

        s3_client.upload_file(local_path, bucket_name, object_key)
        print(f"✅ Preprocessor uploaded to MinIO at s3://{bucket_name}/{object_key}")

    except Exception as e:
        print(f"❌ Failed to upload preprocessor to MinIO: {e}")
    
    # Guardar datos procesados a archivos temporales en lugar de XCom
    processed_data_path = '/tmp/X_processed.npz'
    np.savez_compressed(processed_data_path, X_train_processed=X_train_processed.toarray())
    
    y_train_path = '/tmp/y_train.npy'
    np.save(y_train_path, y_train)
    
    # Solo guardar rutas en XCom
    kwargs['ti'].xcom_push(key='processed_data_path', value=processed_data_path)
    kwargs['ti'].xcom_push(key='y_train_path', value=y_train_path)
    
    return {
        'processed_data_path': processed_data_path,
        'y_train_path': y_train_path
    }


def train_models(**kwargs):
    """Entrenar modelos con restricciones de recursos"""
    processed_data_path = kwargs['ti'].xcom_pull(key='processed_data_path')
    y_train_path = kwargs['ti'].xcom_pull(key='y_train_path')
    
    # Cargar desde archivos temporales
    X_train_processed = np.load(processed_data_path)['X_train_processed']
    y_train = np.load(y_train_path)
    
    # Configuración de múltiples modelos ligeros para comparación
    models = {
        'LightGBM': LGBMClassifier(
            random_state=42, 
            n_jobs=1,            # Usar un solo hilo
            n_estimators=10,     # Muy pocos estimadores
            verbose=-1,          # Desactivar salidas verbosas
            max_depth=3,         # Profundidad mínima
            subsample=0.5,       # Usar solo la mitad de las muestras en cada iteración
            colsample_bytree=0.5 # Usar solo la mitad de las características en cada árbol
        ),
        'DecisionTreeClassifier': DecisionTreeClassifier(
            random_state=42,
            max_depth=3,         # Árbol poco profundo
            min_samples_split=10, # Mínimo de muestras para dividir un nodo
            class_weight='balanced' # Manejar desbalances en las clases
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            solver='liblinear',  # Solver rápido y eficiente
            max_iter=100,        # Pocas iteraciones
            C=1.0,               # Parámetro de regularización estándar
            class_weight='balanced', # Manejar desbalances
            n_jobs=1             # Un solo hilo
        )
    }

    def evaluate_model(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }

    set_mlflow_tracking()

    best_model = None
    best_score = 0
    best_model_obj = None
    model_metrics = {} 
    client = mlflow.tracking.MlflowClient()
    
    with mlflow.start_run() as main_run:
        run_id = main_run.info.run_id
        
        for name, model in models.items():
            print(f"Entrenando modelo: {name}")
            with mlflow.start_run(nested=True):
                # Monitoreo de recursos
                print(f"Iniciando entrenamiento de {name} con {MAX_THREADS} hilos")
                
                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_train_processed)
                metrics = evaluate_model(y_train, y_pred)
                
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                mlflow.log_param('model_name', name)
                
                try:
                    mlflow.sklearn.log_model(
                        model, 
                        name,
                        registered_model_name=name
                    )
                    print(f"Modelo {name} registrado exitosamente")
                except Exception as e:
                    print(f"Error al registrar el modelo {name}: {e}")
                
                avg_score = (metrics['accuracy'] + metrics['f1_score']) / 2
                model_metrics[name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = name
                    best_model_obj = model

                print(f"Modelo {name} - Métricas: {metrics}")

        if best_model:
            print(f"Mejor modelo: {best_model} con puntuación: {best_score}")
            
            try:
                versions = client.search_model_versions(f"name='{best_model}'")
                
                if versions and len(versions) > 0:
                    latest_version = max([int(v.version) for v in versions])
                    
                    client.transition_model_version_stage(
                        name=best_model, 
                        version=latest_version, 
                        stage="Production"
                    )
                    
                    for version in versions:
                        if (version.version != str(latest_version) and 
                            version.current_stage == "Production"):
                            client.transition_model_version_stage(
                                name=best_model,
                                version=version.version,
                                stage="Archived"
                            )
                    
                    print(f"Modelo {best_model} versión {latest_version} marcado como Producción")
                    
                else:
                    print(f"No se encontraron versiones para el modelo {best_model}.")
                    
                    try:
                        print(f"Intentando registrar {best_model} manualmente...")
                        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model}"
                        registered_model = mlflow.register_model(model_uri, best_model)
                        print(f"Modelo registrado manualmente: {registered_model.name}, versión: {registered_model.version}")
                        
                        client.transition_model_version_stage(
                            name=best_model,
                            version=registered_model.version,
                            stage="Production"
                        )
                        print(f"Modelo {best_model} versión {registered_model.version} marcado como Producción")
                    except Exception as e:
                        print(f"Error al intentar registro manual: {e}")
                
            except Exception as e:
                print(f"Error al cambiar el estado del modelo: {e}")
    
    # Limpieza de archivos temporales
    for file_path in [processed_data_path, y_train_path]:
        try:
            os.remove(file_path)
            print(f"Archivo temporal {file_path} eliminado")
        except Exception as e:
            print(f"Error al eliminar archivo temporal {file_path}: {e}")
    
    return {
        'best_model': best_model,
        'best_score': best_score,
        'model_metrics': model_metrics
    }


# Configuración del DAG con intervalo menos frecuente
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'diabetes_ml_pipeline_optimized',
    default_args=default_args,
    description='Pipeline de ML optimizado para predicción de readmisión hospitalaria',
    schedule_interval=timedelta(days=7),  # Ejecutar semanalmente en lugar de diariamente
    catchup=False,
    max_active_runs=1
)

set_mlflow_tracking_task = PythonOperator(
    task_id='set_mlflow_tracking',
    python_callable=set_mlflow_tracking,
    dag=dag
)

start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    provide_context=True,
    dag=dag
)

start_pipeline >> set_mlflow_tracking_task >> load_data_task
load_data_task >> preprocess_data_task >> train_models_task