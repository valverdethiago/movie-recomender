import { getMovieById, getMovieImages } from '$lib/api/movieApi';
import type { Movie } from '$types/movie';
import type { PageLoad } from './$types';

export const load: PageLoad = async ({ params }): Promise<{ movie: Movie }> => {
	const rawMovie = await getMovieById(params.id);
	const movie = await getMovieImages(rawMovie);

	return { movie };
};
