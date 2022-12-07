import sqlite3
import csv
from sqlite3 import Error

def create_connection(db_name):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        return conn
    except Error as e:
        print(e)
    return conn

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def insert_data(conn, table, data):
    command = f"INSERT INTO {table} VALUES({''.join(['?,' for _ in range(len(data)-1)]+['?'])})"
    #print(command)
    try:
        c = conn.cursor()
        c.execute(command, tuple(data))
        conn.commit()
    except Error as e:
        print(e)


def read_csv(file_name):
    with open(file_name, newline = '', encoding = 'utf-8') as csvfile:
        n = 0
        rows = csv.reader(csvfile)
        for row in rows:
            if n == 0:
                n += 1
                continue
            yield row

def main():
    database = r".\db\messagesqlite.db"

    sql_create_message_table = """ CREATE TABLE IF NOT EXISTS Messaging_1 (
                                        contact_name text NOT NULL,
                                        app_name text NOT NULL,
                                        message text NOT NULL
                                    ); """

    sql_create_calendar_table = """CREATE TABLE IF NOT EXISTS Calendar_1 (
                                    name text NOT NULL,                                    
                                    location text,  
                                    content text, 
                                    participant text,
                                    date DATE NOT NULL,
                                    time text NOT NULL
                                );"""
    
    sql_create_mail_table = """CREATE TABLE IF NOT EXISTS Mail_1 (
                                    subject text,                                    
                                    content text,  
                                    copy_recipient text, 
                                    recipient text NOT NULL,
                                    sender text NOT NULL
                                );"""

    # create a database connection
    conn = create_connection(database)

    # create tables
    if conn is not None:
        # create message table
        create_table(conn, sql_create_message_table)

        # create calendar table
        create_table(conn, sql_create_calendar_table)

        # create mail table
        create_table(conn, sql_create_mail_table)
    else:
        print("Error! cannot create the database connection.")
    
    c = conn.cursor()
    for row in c.execute('SELECT * FROM Messaging_1 '):
        print(row)
    for row in c.execute('SELECT * FROM Calendar_1 '):
        print(row)
    for row in c.execute('SELECT * FROM Mail_1 '):
        print(row)
    
    return conn
    """"""


if __name__ == "__main__":
    conn = main()

    for data in read_csv('message_entityies.csv'):
        print(data)
        insert_data(conn, 'Messaging_1', data)

    for data in read_csv('events.csv'):
        print(data)
        insert_data(conn, 'Calendar_1', data)

    for data in read_csv('mail_entityies.csv'):
        print(data)
        insert_data(conn, 'Mail_1', data)
