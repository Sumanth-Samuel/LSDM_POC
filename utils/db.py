# utils/db.py
import pyodbc
import pandas as pd
import os

conn_str = os.getenv("SQL_CONN_STRING")

def run_sql_query(sql: str):
    try:
        conn = pyodbc.connect(conn_str)
        df = pd.read_sql(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, str(e)
