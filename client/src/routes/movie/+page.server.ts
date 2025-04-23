import { getMovieImages, getSuggestedMovies } from '$lib/api/movieApi';
import type { Movie } from '$types/movie';
import { redirect } from '@sveltejs/kit';
import type { PageServerLoad } from '../$types';

export const load: PageServerLoad = async ({ cookies }): Promise<{ movies: Movie[] }> => {
	const authCookie = cookies.get('auth');

	if (!authCookie) {
		return redirect(302, '/');
	}

	const suggestedMovies = await getSuggestedMovies(authCookie);

	const promisesArray = suggestedMovies.map((movies) => getMovieImages(movies));

	const movies = await Promise.all(promisesArray);

	return { movies };
};
