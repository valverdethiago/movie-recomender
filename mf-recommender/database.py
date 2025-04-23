import duckdb
import csv
import ast  # To safely parse string representations of lists/dicts
import os
from faker import Faker
import time
from datetime import datetime
import json

DATABASE_FILE = "movies.db"
RATINGS_FILE = "the-movies-dataset/ratings_small.csv"
MOVIES_FILE = "the-movies-dataset/movies_metadata.csv"

CREATE_USERS_TABLE_SQL = """
    CREATE TABLE Users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT
    )
"""

CREATE_WATCHED_TABLE_SQL = """
    CREATE TABLE Watched (
        user_id INTEGER,
        movie_id INTEGER,
        PRIMARY KEY (user_id, movie_id),
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
"""

CREATE_MOVIES_TABLE_SQL = """
        CREATE TABLE movies (
            id BIGINT PRIMARY KEY,
            original_title TEXT,
            year INT,
            poster_path TEXT,
            backdrop_path TEXT,
            imdb_id TEXT,
            description TEXT,
        )
    """

CREATE_GENRES_TABLE_SQL = """
        CREATE TABLE genres (
            movie_id BIGINT,
            genre_id INT,
            genre_name TEXT
        )
    """

DROP_TABLE_IF_EXISTS_SQL = "DROP TABLE IF EXISTS {table}"
DROP_MOVIES_TABLE_IF_EXISTS_SQL = DROP_TABLE_IF_EXISTS_SQL.format(table="movies")
DROP_GENRES_TABLE_IF_EXISTS_SQL = DROP_TABLE_IF_EXISTS_SQL.format(table="genres")
DROP_USERS_TABLE_IF_EXISTS_SQL = DROP_TABLE_IF_EXISTS_SQL.format(table="users")
DROP_WATCHED_TABLE_IF_EXISTS_SQL = DROP_TABLE_IF_EXISTS_SQL.format(table="watched")

def time_elapsed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Capture the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Time elapsed for {func.__name__}: {elapsed_time:.4f} seconds")
        return result
    return wrapper

def extract_year_safe(date_str, default=None):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").year
    except ValueError:
        return default

def extract_paths(json_string):
    try:
        data = json.loads(json_string.replace("'", '"'))  # Convert single quotes to double quotes
        poster_path = data.get('poster_path', '')
        backdrop_path = data.get('backdrop_path', '')
        return poster_path, backdrop_path
    except json.JSONDecodeError:
        return "", ""

@time_elapsed
def load_movies_and_genres():
    
    conn = duckdb.connect(DATABASE_FILE)
    
    conn.execute(DROP_MOVIES_TABLE_IF_EXISTS_SQL)
    conn.execute(DROP_GENRES_TABLE_IF_EXISTS_SQL)

    conn.execute(CREATE_MOVIES_TABLE_SQL)
    conn.execute(CREATE_GENRES_TABLE_SQL)

    with open(MOVIES_FILE, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for line_number, row in enumerate(reader, start=2):
            try:
                movie_id = row.get("id", "").strip()
                title = row.get("original_title", "").strip()
                release_date = row.get("release_date", "").strip()
                belongs_to_collection = row.get("belongs_to_collection", "").strip()
                imdb_id = row.get("imdb_id", "").strip()
                description = row.get("overview", "").strip()

                if not movie_id.isdigit():
                    continue

                movie_id = int(movie_id)

                year = extract_year_safe(release_date)

                poster, backdrop = extract_paths(belongs_to_collection)
                
                conn.execute(
                    """
                        INSERT OR IGNORE 
                        INTO movies (id, original_title, year,  poster_path, backdrop_path, imdb_id, description) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, 
                    (movie_id, title if title else "Unknown", year, poster, backdrop, imdb_id, description)
                )

                genres_data = row.get("genres", "[]").strip()
                try:
                    genres_list = ast.literal_eval(genres_data)
                except (SyntaxError, ValueError):
                    print(f"Error on line {line_number}: {e}")
                    continue


                for genre in genres_list:
                    genre_id = genre.get("id", -1)
                    genre_name = genre.get("name", "Unknown")
                    conn.execute("INSERT INTO genres (movie_id, genre_id, genre_name) VALUES (?, ?, ?)", (movie_id, genre_id, genre_name))
            except Exception as e:
                print(f"Error on line {line_number}: {e}")
    
    conn.close()

@time_elapsed
def load_users_and_watched():

    conn = duckdb.connect(DATABASE_FILE)
    
    conn.execute(DROP_WATCHED_TABLE_IF_EXISTS_SQL)
    conn.execute(DROP_USERS_TABLE_IF_EXISTS_SQL)

    conn.execute(CREATE_USERS_TABLE_SQL)
    conn.execute(CREATE_WATCHED_TABLE_SQL)

    fake = Faker()
    
    with open(RATINGS_FILE, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            user_id, movie_id, _, _ = row  
            user_id = int(user_id)
            movie_id = int(movie_id)
            user_name = fake.name()
            user_email = fake.email()
            conn.execute("INSERT OR IGNORE INTO Users (id, name, email) VALUES (?, ?, ?)", [user_id, user_name, user_email])
            conn.execute("INSERT INTO Watched (user_id, movie_id) VALUES (?, ?)", [user_id, movie_id])
    conn.close()

# Load movies and watches for aditional users
@time_elapsed
def load_users_and_watched_additional():
    """
    Load aditional users for further avaluation in the app.
    """

    conn = duckdb.connect(DATABASE_FILE)

    # Load CSV and insert into an existing table
    conn.execute("INSERT INTO my_table SELECT * FROM read_csv_auto('ratings_additional.csv')")

    conn.close()

def load_database(force_load=False):
    if os.path.exists(DATABASE_FILE) and not force_load:
        print(f"Database {DATABASE_FILE} already exists. Skipping loading.")
        return
    
    load_users_and_watched()
    load_movies_and_genres()
    


def get_user_data(user_id):
    conn = duckdb.connect(DATABASE_FILE)
    user_data = conn.execute("SELECT * FROM users WHERE id = ?", [user_id]).fetchall()
    if not user_data:
        return None
    user_data_dict = {
        "userId": user_data[0][0],
        "name": user_data[0][1],
        "email": user_data[0][2]
    }

    watched_movies = conn.execute("""
        SELECT m.original_title 
        FROM watched w
        JOIN movies m ON w.movie_id = m.id
        WHERE w.user_id = ?
    """, [user_id]).fetchall()
    user_data_dict["watched_movies"] = [movie[0] for movie in watched_movies]

    conn.close()
    return user_data_dict

def get_user_watched_movies(user_id):
    """
    Get a list of watched movie titles and their genres for a given user.
    """
    conn = duckdb.connect(DATABASE_FILE)
    watched_movies = conn.execute("""
        SELECT m.id, m.original_title, STRING_AGG(g.genre_name, ', ') AS genres, m.year
        FROM watched w
        JOIN movies m ON w.movie_id = m.id
        LEFT JOIN genres g ON m.id = g.movie_id
        WHERE w.user_id = ?
        GROUP BY m.id, m.original_title, m.year
        ORDER BY m.year desc
    """, [user_id]).fetchall()
    conn.close()
    
    if not watched_movies:
        return None
    
    return [{"movie_id": movie[0], "title": movie[1], "genres": movie[2], "year": movie[3]} 
            for movie in watched_movies]

def get_movie_data(movie_id):
    conn = duckdb.connect(DATABASE_FILE)
    movie_data = conn.execute("SELECT * FROM movies WHERE id = ?", [movie_id]).fetchall()
    if not movie_data:
        return None
    
    movie_data_dict = {
        "id": movie_data[0][0],
        "original_title": movie_data[0][1],
        "year": movie_data[0][2],
        "poster_path": movie_data[0][3],
        "backdrop_path": movie_data[0][4],
        "imdb_id": movie_data[0][5],
        "description": movie_data[0][6],
    }

    genres = conn.execute("SELECT genre_name FROM genres WHERE movie_id = ?", [movie_id]).fetchall()
    movie_data_dict["genres"] = [genre[0] for genre in genres]

    conn.close()
    return movie_data_dict


def search_movies_by(keywords):
    conn = duckdb.connect(DATABASE_FILE)
    # Prepare the query with a case-insensitive search
    query_movies = """
    SELECT 
        m.id, 
        original_title, 
        GROUP_CONCAT(genre_name) AS genres, 
        m.year,
    FROM movies m JOIN genres g ON m.id = g.movie_id 
    WHERE LOWER(m.original_title) LIKE ?
    GROUP BY m.id, m.original_title, m.year
    ORDER BY m.year desc
    """
    filters = [f"%{'%'.join(keywords).lower()}%"]
    
    # Execute the query and fetch movie results
    movie_results = conn.execute(query_movies, filters).fetchall()

    # Fetch watched count for each movie separately
    movies = []
    for movie_id, title, genres, year in movie_results:
        query_watched_count = "SELECT COUNT(DISTINCT user_id) FROM watched WHERE movie_id = ?"
        watched_count = conn.execute(query_watched_count, [movie_id]).fetchone()[0]

        movies.append({
            "movie_id": movie_id,
            "title": title,
            "genres": genres,
            "year": year,
            "watched_count": watched_count
        })

    return movies
