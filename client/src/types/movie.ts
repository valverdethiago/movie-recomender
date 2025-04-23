export type Movie = {
	movie_id: number;
	title?: string;
	original_title?: string;
	poster_path?: string;
	backdrop_path?: string;
	description: string;
	genres: string[];
	year: number;
	imdb_id?: string;
};
