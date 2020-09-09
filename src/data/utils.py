from pyspark.sql import SparkSession

def start_session(app_name, data_path):
    """
    Start PySpark session and return dataframe of raw data.
    
    Arguments:
        app_name (str) - Name of Spark session app for use in Spark Web UI
        data_path (str) - Path to original csv containing raw data

    Returns:
        df (pyspark.sql.dataframe.DataFrame) - Spark dataframe containing raw data
        spark (pyspark.sql.session.) - Spark session required for queries
    """
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    df = spark.read.csv(data_path, header="true")
    return df, spark

def save_data(df, path):
    """
    Save PySpark DataFrame as csv.

    Arguments:
        df (pyspark.sql.dataframe.DataFrame) - Result of PySpark SQL query
        path (str) - Path to save data
    """
    df.coalesce(1).write.format('csv').save(path, header='true')