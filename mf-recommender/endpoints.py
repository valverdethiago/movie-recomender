from flask import Blueprint, jsonify
from database import get_user_data, get_user_watched_movies, get_movie_data, search_movies_by

endpoints = Blueprint("endpoints", __name__)

# API Endpoint 1: Get all data of a particular user ID
@endpoints.route("/user/<int:user_id>", methods=["GET"])
def get_user_data_endpoint(user_id):
    user_data = get_user_data(user_id)
    if not user_data:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user_data)

# API Endpoint 2: Get movies watched by a user ID
@endpoints.route("/user/<int:user_id>/movies", methods=["GET"])
def get_user_watched_movies_endpoint(user_id):
    watched_movies = get_user_watched_movies(user_id)
    if not watched_movies:
        return jsonify({"error": "No watched movies found for this user"}), 404
    return jsonify(watched_movies)

# API Endpoint 3: Get all data of a particular movie ID
@endpoints.route("/movie/<int:movie_id>", methods=["GET"])
def get_movie_data_endpoint(movie_id):
    movie_data = get_movie_data(movie_id)
    if not movie_data:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(movie_data)

@endpoints.route('/search_movies/<keywords>', methods=['GET'])
def search_movies(keywords):
    """ Endpoint to search for movies based on a set of keywords passed in the URL """
    # Split the keywords by space
    keyword_list = keywords.split(',')

    if not keyword_list:
        return jsonify({"error": "No keywords provided"}), 400
    
    movies = search_movies_by(keyword_list)
    
    return jsonify(movies)
