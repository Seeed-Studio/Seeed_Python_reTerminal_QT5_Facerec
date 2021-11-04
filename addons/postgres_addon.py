import psycopg2
import psycopg2.extras
import numpy as np
import base64

def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except psycopg2.OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_write_query(connection, query):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Query executed successfully")
    except psycopg2.OperationalError as e:
        print(f"The error '{e}' occurred")

def execute_read_query(connection, query):
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except psycopg2.OperationalError as e:
        print(f"The error '{e}' occurred")

def read_db():
    connection = create_connection("face_rec", "postgres", "NULL", "127.0.0.1", "5432")
    select_users = "SELECT * FROM users"
    users = execute_read_query(connection, select_users)
    dict_db = {}

    for user in users:
        user['vector'] = np.frombuffer(base64.b64decode(user['vector']), np.float32)
        print(f"User ID: {user['id']} Name: {user['name']}")
        print("Face features vector: \n", user['vector'])
        dict_db[user['id']] = {'name': user['name'], 'vector': user['vector']}

    return dict_db

def add_entry(id, name, feature_array):
    connection = create_connection("face_rec", "postgres", "NULL", "127.0.0.1", "5432")    
    vector = base64.b64encode(feature_array).decode('utf-8')

    connection.autocommit = True
    cursor = connection.cursor()
    cursor.execute(
        """INSERT INTO users (id, name, vector) VALUES (%s, %s, %s)
        ON CONFLICT (id) 
        DO UPDATE SET name = %s, vector = %s""", (id, name, vector,
                                                name, vector))

def delete_entry(id):
    connection = create_connection("face_rec", "postgres", "NULL", "127.0.0.1", "5432") 
    delete_user = f"DELETE FROM users WHERE id = {id}"
    execute_write_query(connection, delete_user)