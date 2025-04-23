import type { Movie } from '$types/movie';
import type { TmdbSearchResults } from '$types/tmdb';

const API_URL = 'http://localhost:5000';
const TMDB_URL = 'https://api.themoviedb.org/3';
const TMDB_API_KEY = import.meta.env.VITE_TMDB_TOKEN;

export const getSuggestedMovies = async (userId: string): Promise<Movie[]> => {
	const response = await fetch(`${API_URL}/recommendations/${userId}`);
	if (!response.ok) {
		throw new Error('Failed to fetch movies');
	}
	return response.json();
};

export const getMovieById = async (id: string): Promise<Movie> => {
	const response = await fetch(`${API_URL}/movie/${id}`);
	if (!response.ok) {
		throw new Error('Movie not found');
	}
	return response.json();
};

export const getMovieImages = async (movie: Movie): Promise<Movie> => {
	const response = await fetch(`${TMDB_URL}/find/${movie.imdb_id}?external_source=imdb_id`, {
		headers: {
			accept: 'application/json',
			Authorization: `Bearer ${TMDB_API_KEY}`
		}
	});
	if (!response.ok) {
		throw new Error('Failed to fetch movie images');
	}
	const data = (await response.json()) as TmdbSearchResults;

	if (data.movie_results.length > 0) {
		const tmdbMovie = data.movie_results[0];
		movie.poster_path = tmdbMovie.poster_path;
		movie.backdrop_path = tmdbMovie.backdrop_path;
	}

	return movie;
};
