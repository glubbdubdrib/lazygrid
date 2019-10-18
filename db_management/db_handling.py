# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 22:39:52 2019

@author: Sam
"""

import sqlite3
import os
from sqlite3 import Error

def init_database(directory):
    
    #Creating a directory where the database will be located

    if not os.path.isdir(directory):
        print("Creating directory")
        os.mkdir(directory)
        
    #Connecting to the database
    db = sqlite3.connect(directory + '/weighs_as_joblibs.db')
    cur = db.cursor()
    try:
        cur.execute('CREATE TABLE joblibs (id text primary key, joblib blob)')
    except:
        cur.close()
        return db
    cur.close()
    return db
        
def db_add_row(db, identifier, filepath):
    
    cur = db.cursor()
    cur.execute("INSERT INTO joblibs (id, joblib) VALUES (?,?)", (identifier, convert_to_binary(filepath)))
    cur.close()
    db.commit()

def db_get_row(db, identifier):
    identifier = identifier.split('.')[0].split('/')[1]
    cur = db.cursor()
    cur.execute("SELECT * FROM joblibs WHERE id = ?", (identifier,))
    temp = cur.fetchall()
    cur.close()
    return temp

def convert_to_binary(filepath):

    with open(filepath,'rb') as pickled_file:
        binary_data = pickled_file.read()
        pickled_file.close()
        return binary_data

if __name__ == "__main__":
    init_database("output")