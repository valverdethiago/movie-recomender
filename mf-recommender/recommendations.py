import duckdb
import pickle
from flask import Blueprint, request, jsonify

recommendations_bp = Blueprint('recommendations', __name__)

# Load recommendations from a pickle file
PICKLE_FILE = "recs.pkl"
try:
    with open(PICKLE_FILE, "rb") as f:
        recommendations = pickle.load(f)
except FileNotFoundError:
    recommendations = {}

def get_recommendations(user_id, db_path="movies.db"):
    """
    Get recommended movies for a user, including title and genres, excluding already watched movies.
    
    :param user_id: The user ID to fetch recommendations for.
    :param db_path: Path to the DuckDB database file.
    :return: A list of up to 15 recommended movies with title and genres as dictionaries.
    """
    if user_id not in recommendations:
        return []
    
    recommended_movies = [movie_id for movie_id, _ in recommendations[user_id]]
    
    conn = duckdb.connect(db_path)
    
    # Get watched movies for the user
    watched_query = """
        SELECT movie_id FROM watched WHERE user_id = ?
    """
    watched_movies = set(conn.execute(watched_query, [user_id]).fetchall())
    
    # Filter out watched movies
    filtered_movies = [movie for movie in recommended_movies if (movie,) not in watched_movies]
    
    if not filtered_movies:
        return []
    
    # Get movie details
    movie_query = """
    SELECT 
        m.id, 
        m.original_title, 
        STRING_AGG(g.genre_name, ', ') AS genres, 
        m.year, 
        m.poster_path, 
        m.backdrop_path,
        m.imdb_id,
        m.description
    FROM movies m
    LEFT JOIN genres g ON m.id = g.movie_id
    WHERE m.id IN ({})
    GROUP BY 
        m.id, 
        m.original_title, 
        m.year, 
        m.poster_path, 
        m.backdrop_path,
        m.imdb_id,
        m.description
    ORDER BY m.year DESC
    LIMIT 10;
    """.format(','.join('?' * len(filtered_movies)))
    
    movie_details = conn.execute(movie_query, filtered_movies).fetchall()
    
    conn.close()
    
    return [
                {
                    "movie_id": movie_id, 
                    "title": title, 
                    "genres": genres, 
                    "year":year, 
                    "poster_path": poster_path, 
                    "backdrop_path": backdrop_path, 
                    "imdb_id": imdb_id,
                    "description": description,
                }
            for movie_id, title, genres, year, poster_path, backdrop_path, imdb_id, description in movie_details]

@recommendations_bp.route('/recommendations/<user_id>', methods=['GET'])
def recommendations_endpoint(user_id):
    user_id = int(user_id)
    recommended_movies = get_recommendations(user_id)
    return jsonify(recommended_movies)
