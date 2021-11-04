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

def create_database(connection, query):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Query executed successfully")
    except psycopg2.OperationalError as e:
        print(f"The error '{e}' occurred")
    except psycopg2.ProgrammingError as ee:
        print("exists")

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


connection = create_connection("postgres", "postgres", "NULL", "127.0.0.1", "5432")

create_database_query = "CREATE DATABASE face_rec"
create_database(connection, create_database_query)

connection = create_connection("face_rec", "postgres", "NULL", "127.0.0.1", "5432")

create_users_table = """
CREATE TABLE IF NOT EXISTS users (
  id INT PRIMARY KEY,
  name TEXT NOT NULL, 
  vector TEXT
)
"""

execute_write_query(connection, create_users_table)

some_array = np.random.rand(1, 128).astype('float32')
print(some_array.shape)
print(some_array)
vector = base64.b64encode(some_array).decode('utf-8')
user = {"id": 0, "name": "James Doe", "vector": vector}

connection.autocommit = True
cursor = connection.cursor()
cursor.execute(
    """INSERT INTO users (id, name, vector) VALUES (%s, %s, %s)
    ON CONFLICT (id) 
    DO UPDATE SET name = %s, vector = %s""", (user['id'],user['name'],user['vector'],
                                              user['name'], user['vector']))

select_users = "SELECT * FROM users"
users = execute_read_query(connection, select_users)

for user in users:
    user['vector'] = np.frombuffer(base64.b64decode(user['vector']), np.float32)
    print(user['vector'].shape)
    print(f"User ID: {user['id']} Name: {user['name']}")
    print("Face features vector: \n", user['vector'])

#id = 0
#delete_user = f"DELETE FROM users WHERE id = {id}"
#execute_write_query(connection, delete_user)
#print(users)