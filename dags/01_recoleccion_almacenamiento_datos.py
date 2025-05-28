import os
import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.model_selection import train_test_split

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'diabetes_data_processing',
    default_args=default_args,
    description='Descargar, procesar y almacenar datos de diabetes',
    schedule_interval=timedelta(days=1,hours=1), # un dia y una hora
    start_date=datetime(2025, 4, 1),
    catchup=False,
    max_active_runs=1, 
    tags=['diabetes', 'procesamiento-datos'],
)

TEMP_DIR = '/opt/airflow/temp'

BATCH_SIZE = 15000

def create_temp_directory():
    """
    Crea un directorio temporal para los archivos intermedios
    que se usarán durante el procesamiento.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    return "Directorio temporal creado exitosamente"

# Descargar el conjunto de datos
def download_data():
    """
    Descarga el conjunto de datos de diabetes desde la URL proporcionada.
    Retorna la ruta del archivo temporal descargado.
    """
    url = 'https://docs.google.com/uc?export=download&id=1k5-1caezQ3zWJbKaiMULTGq-3sz6uThC'
    temp_data_path = os.path.join(TEMP_DIR, 'diabetes_temp.csv')
    
    print(f"Descargando conjunto de datos a {temp_data_path}")
    r = requests.get(url, allow_redirects=True, stream=True)
    with open(temp_data_path, 'wb') as f:
        f.write(r.content)
    
    return temp_data_path

def store_raw_data_in_postgres(ti):
    """
    Almacena los datos crudos (sin ningún procesamiento) en la base de datos PostgreSQL.
    Crea el esquema y la tabla si no existen.
    
    Args:
        ti: Instancia de XCom para obtener la ruta de los datos crudos
        
    Returns:
        Un mensaje indicando que los datos crudos se han almacenado
    """
    raw_data_path = ti.xcom_pull(task_ids='download_data')
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    pg_hook.run("""
    CREATE SCHEMA IF NOT EXISTS raw_data;
    
    DROP TABLE IF EXISTS raw_data.diabetes;
    
    CREATE TABLE IF NOT EXISTS raw_data.diabetes (
        id SERIAL PRIMARY KEY,
        encounter_id INTEGER,
        patient_nbr BIGINT,
        race VARCHAR(50),
        gender VARCHAR(20),
        age VARCHAR(20),
        weight VARCHAR(20),
        admission_type_id INTEGER,
        discharge_disposition_id INTEGER,
        admission_source_id INTEGER,
        time_in_hospital INTEGER,
        payer_code VARCHAR(20),
        medical_specialty VARCHAR(100),
        num_lab_procedures INTEGER,
        num_procedures INTEGER,
        num_medications INTEGER,
        number_outpatient INTEGER,
        number_emergency INTEGER,
        number_inpatient INTEGER,
        diag_1 VARCHAR(50),
        diag_2 VARCHAR(50),
        diag_3 VARCHAR(50),
        number_diagnoses INTEGER,
        max_glu_serum VARCHAR(50),
        a1cresult VARCHAR(50),
        metformin VARCHAR(50),
        repaglinide VARCHAR(50),
        nateglinide VARCHAR(50),
        chlorpropamide VARCHAR(50),
        glimepiride VARCHAR(50),
        acetohexamide VARCHAR(50),
        glipizide VARCHAR(50),
        glyburide VARCHAR(50),
        tolbutamide VARCHAR(50),
        pioglitazone VARCHAR(50),
        rosiglitazone VARCHAR(50),
        acarbose VARCHAR(50),
        miglitol VARCHAR(50),
        troglitazone VARCHAR(50),
        tolazamide VARCHAR(50),
        examide VARCHAR(50),
        citoglipton VARCHAR(50),
        insulin VARCHAR(50),
        "glyburide-metformin" VARCHAR(50),
        "glipizide-metformin" VARCHAR(50),
        "glimepiride-pioglitazone" VARCHAR(50),
        "metformin-rosiglitazone" VARCHAR(50),
        "metformin-pioglitazone" VARCHAR(50),
        change VARCHAR(50),
        diabetesmed VARCHAR(20),
        readmitted VARCHAR(20),
        dataset VARCHAR(20) DEFAULT 'raw'
    );
    """)
    
    df = pd.read_csv(raw_data_path)
    
    temp_csv_path = os.path.join(TEMP_DIR, 'temp_raw_diabetes.csv')
    df.to_csv(temp_csv_path, index=False) 
    
    conn = pg_hook.get_conn()
    cur = conn.cursor()
    
    with open(temp_csv_path, 'r') as f:
        next(f) 
        cur.copy_expert(
            sql="""COPY raw_data.diabetes (
                encounter_id, patient_nbr, race, gender, age, weight, 
                admission_type_id, discharge_disposition_id, admission_source_id, 
                time_in_hospital, payer_code, medical_specialty, num_lab_procedures, 
                num_procedures, num_medications, number_outpatient, number_emergency, 
                number_inpatient, diag_1, diag_2, diag_3, number_diagnoses, max_glu_serum, 
                a1cresult, metformin, repaglinide, nateglinide, chlorpropamide, 
                glimepiride, acetohexamide, glipizide, glyburide, tolbutamide, 
                pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, 
                tolazamide, examide, citoglipton, insulin, "glyburide-metformin", 
                "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", 
                "metformin-pioglitazone", change, diabetesmed, readmitted) 
                FROM STDIN WITH CSV""",
            file=f  
        )
    
    conn.commit()
    
    os.remove(temp_csv_path)
    
    return "Datos crudos almacenados en PostgreSQL sin ningún procesamiento"

def process_data(ti):
    """
    Procesa los datos crudos realizando limpieza y transformaciones.
    
    Argumentos para experimentación:
    - Se manejan valores faltantes: '?' se reemplazan con NaN y luego se imputan
    - Para variables numéricas: se imputan con la mediana y se manejan outliers
    - Para variables categóricas: se imputan con el modo o valores predeterminados
    - Se realizan codificaciones one-hot para variables categóricas seleccionadas
    - Se crean características adicionales como el total de medicamentos
    - Se eliminan columnas redundantes o de baja utilidad predictiva
    
    Args:
        ti: Instancia de XCom para obtener la ruta de los datos crudos
        
    Returns:
        La ruta del archivo temporal procesado
    """
    # raw_data_path = ti.xcom_pull(task_ids='download_data')
    # print(f"Procesando datos de {raw_data_path}")
    
    # df = pd.read_csv(raw_data_path)
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    df = pg_hook.get_pandas_df("SELECT * FROM raw_data.diabetes WHERE dataset = 'raw'")
    print("Columnas dataframe a procesar",df.columns)
    print(f"Dimensiones originales: {df.shape}")
    
    df = df.replace(['?', '', 'None', 'NULL'], np.nan)
    
    num_duplicates = df.duplicated(subset=['encounter_id']).sum()
    if num_duplicates > 0:
        print(f"Eliminando {num_duplicates} registros duplicados")
        df = df.drop_duplicates(subset=['encounter_id'])
    
    print(f"Valores únicos en race: {df['race'].unique()}")
    df['race'] = df['race'].fillna('Unknown')
    
    print(f"Valores únicos en gender: {df['gender'].unique()}")
    df['gender'] = df['gender'].fillna('Unknown')
    
    print(f"Valores únicos en age: {df['age'].unique()}")
    df['age'] = df['age'].fillna('Unknown')
    
    numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                   'num_medications', 'number_outpatient', 'number_emergency', 
                   'number_inpatient', 'number_diagnoses']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        
        print(f"Estadísticas para {col}: Media={mean_val}, STD={std_val}, Min={min_val}, Max={max_val}")
        
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val
        
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        df[diag_col] = df[diag_col].fillna('Unknown')
        
        invalid_codes = df[df[diag_col].str.contains('[^0-9.-]', na=False, regex=True)][diag_col].unique()
        if len(invalid_codes) > 0:
            print(f"Códigos de diagnóstico inválidos en {diag_col}: {invalid_codes}")
    
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'examide',
                'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
    
    for med_col in med_cols:
        unique_vals = df[med_col].unique()
        print(f"Valores únicos en {med_col}: {unique_vals}")
        
        valid_vals = ['No', 'Up', 'Down', 'Steady']
        df[med_col] = df[med_col].apply(lambda x: x if x in valid_vals else 'No')
        
        df[med_col] = df[med_col].fillna('No')
    
    print(f"Valores únicos en readmitted: {df['readmitted'].unique()}")
    df['readmitted'] = df['readmitted'].fillna('NO')
   
   
    print(f"Dimensiones después de la limpieza: {df.shape}")
    print(f"Valores faltantes por columna:\n{df.isnull().sum()}")

    int_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                  'num_medications', 'number_outpatient', 'number_emergency', 
                  'number_inpatient', 'number_diagnoses', 'total_diabetes_meds']
    
    if 'changed_med' in df.columns:
        int_columns.append('changed_med')
            
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    
    processed_path = os.path.join(TEMP_DIR, 'processed_diabetes_temp.csv')
    df.to_csv(processed_path, index=False)
    
    return processed_path

def split_and_load_data(ti):
    """
    Divide los datos procesados en conjuntos de entrenamiento, validación y prueba
    y los carga directamente en PostgreSQL.
    
    Argumentos para experimentación:
    - Proporción de datos: 70% entrenamiento, 15% validación, 15% prueba
    - Semilla aleatoria: 42 para garantizar reproducibilidad
    - Tamaño de lote: 15,000 registros para entrenar por lotes (requisito del proyecto)
    
    """
    processed_path = ti.xcom_pull(task_ids='process_data')
    print(f"Dividiendo y cargando datos de {processed_path}")
    
    df = pd.read_csv(processed_path)
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = pg_hook.get_conn()
    
    pg_hook.run("""
    CREATE SCHEMA IF NOT EXISTS clean_data;
    
    -- Crear tablas para datos de entrenamiento, validación y prueba
    DROP TABLE IF EXISTS clean_data.diabetes_train;
    DROP TABLE IF EXISTS clean_data.diabetes_validation;
    DROP TABLE IF EXISTS clean_data.diabetes_test;
    """)
    
    # ⚠️ Evitar incluir columnas que se declaran manualmente después
    excluded_columns = {'id', 'batch_id', 'dataset'}  # ya se agregan abajo manualmente

    column_definitions = []
    for column in df.columns:
        if column in excluded_columns:
            continue
        if column in ['encounter_id', 'time_in_hospital', 'num_lab_procedures', 
                    'num_procedures', 'num_medications', 'number_outpatient', 
                    'number_emergency', 'number_inpatient', 'number_diagnoses', 
                    'total_diabetes_meds', 'changed_med']:
            column_definitions.append(f'"{column}" INTEGER')
        elif column.startswith(('gender_', 'age_', 'max_glu_serum_', 'a1cresult_', 'diabetesmed_')):
            column_definitions.append(f'"{column}" INTEGER')
        else:
            column_definitions.append(f'"{column}" VARCHAR(100)')

    
    for dataset in ['train', 'validation', 'test']:
        pg_hook.run(f"""
        CREATE TABLE IF NOT EXISTS clean_data.diabetes_{dataset} (
            {', '.join(column_definitions)},
            batch_id INTEGER,
            dataset VARCHAR(20) DEFAULT '{dataset}'
        );
        """)

    def drop_if_exists(df, columns):
        return df.drop(columns=[col for col in columns if col in df.columns], errors='ignore')

    val_df = drop_if_exists(val_df, ['id'])
   
       
    temp_val_path = os.path.join(TEMP_DIR, 'temp_validation.csv')
    val_df.to_csv(temp_val_path, index=False)
    
    column_names = ','.join([f'"{col}"' for col in val_df.columns])
    
    cur = conn.cursor()
    with open(temp_val_path, 'r') as f:
        next(f)  
        cur.copy_expert(
            sql=f"""COPY clean_data.diabetes_validation ({column_names}) FROM STDIN WITH CSV""",
            file=f
        )
    conn.commit()
    
    test_df = drop_if_exists(test_df, ['id'])

    temp_test_path = os.path.join(TEMP_DIR, 'temp_test.csv')
    test_df.to_csv(temp_test_path, index=False)
    
    column_names = ','.join([f'"{col}"' for col in test_df.columns])
    
    cur = conn.cursor()
    with open(temp_test_path, 'r') as f:
        next(f)  
        cur.copy_expert(
            sql=f"""COPY clean_data.diabetes_test ({column_names}) FROM STDIN WITH CSV""",
            file=f
        )
    conn.commit()
    
    num_batches = (len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE 
    
    for i in range(num_batches):
        batch_id = i + 1
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(train_df))
        
        batch_df = train_df.iloc[start_idx:end_idx]
        
        batch_df = drop_if_exists(batch_df, ['id'])

        temp_batch_path = os.path.join(TEMP_DIR, f'temp_train_batch_{batch_id}.csv')
        batch_df.to_csv(temp_batch_path, index=False)
        
        column_names = ','.join([f'"{col}"' for col in batch_df.columns])
        
        cur = conn.cursor()
        with open(temp_batch_path, 'r') as f:
            next(f)  
            cur.copy_expert(
                sql=f"""COPY clean_data.diabetes_train ({column_names}) FROM STDIN WITH CSV""",
                file=f
            )
        conn.commit()
        
        pg_hook.run(f"UPDATE clean_data.diabetes_train SET batch_id = {batch_id} WHERE batch_id IS NULL")
        
        os.remove(temp_batch_path)
    
    os.remove(temp_val_path)
    os.remove(temp_test_path)
    
    pg_hook.run("""
    DROP TABLE IF EXISTS clean_data.batch_info;
    
    CREATE TABLE IF NOT EXISTS clean_data.batch_info (
        batch_id INTEGER PRIMARY KEY,
        batch_size INTEGER,
        creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
        
    for i in range(num_batches):
        batch_id = i + 1
        batch_size = min(BATCH_SIZE, len(train_df) - i * BATCH_SIZE)
        
        pg_hook.run(f"""
        INSERT INTO clean_data.batch_info (batch_id, batch_size)
        VALUES ({batch_id}, {batch_size});
        """)
    
    return f"Datos divididos y cargados en PostgreSQL: {num_batches} lotes de entrenamiento, {len(val_df)} validación, {len(test_df)} prueba"

def cleanup_temp_files(ti):
    """
    Limpia los archivos temporales creados durante el proceso.
    
    Args:
        ti: Instancia de XCom

    """
    try:
        raw_data_path = ti.xcom_pull(task_ids='download_data')
        if os.path.exists(raw_data_path):
            os.remove(raw_data_path)
        
        processed_path = ti.xcom_pull(task_ids='process_data')
        if os.path.exists(processed_path):
            os.remove(processed_path)
        
        for filename in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return "Archivos temporales limpiados exitosamente"
    except Exception as e:
        return f"Error al limpiar archivos temporales: {str(e)}"

create_temp_dir_task = PythonOperator(
    task_id='create_temp_directory',
    python_callable=create_temp_directory,
    dag=dag,
)

download_task = PythonOperator(
    task_id='download_data',
    python_callable=download_data,
    dag=dag,
)

store_raw_task = PythonOperator(
    task_id='store_raw_data',
    python_callable=store_raw_data_in_postgres,
    dag=dag,
)

process_task = PythonOperator(
    task_id='process_data',
    python_callable=process_data,
    dag=dag,
)

split_and_load_task = PythonOperator(
    task_id='split_and_load_data',
    python_callable=split_and_load_data,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_temp_files',
    python_callable=cleanup_temp_files,
    dag=dag,
)

create_temp_dir_task >> download_task >> store_raw_task >> process_task >> split_and_load_task >> cleanup_task