import utils
import os

#Set up PySpark environment
path = '/home/gerardo/Desktop/Projects/Datasets/PGA/PGA_Data_Historical.csv'
raw_data, spark = utils.start_session(app_name='Driving-Distance', data_path=path)
raw_data.createOrReplaceTempView("raw_data")

#Get average driving distance and date of each tournament in chronological order
query = "SELECT tournament, date, AVG(value) AS average_driving_distance\
         FROM (SELECT tournament, date, value FROM raw_data WHERE statistic='Driving Distance' AND variable='AVG.')\
         GROUP BY tournament, date\
         ORDER BY date ASC"
data = spark.sql(query)
utils.save_data(data, 'avg-driving-distance')