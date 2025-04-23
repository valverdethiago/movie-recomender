<script lang="ts">
	import type { Movie } from '$types/movie';
	export let movie: Movie;
</script>

<a
	href={`/movie/${movie.movie_id}`}
	class="block rounded-lg shadow transition hover:shadow-lg movie-preview"
>
	<img
		src={`https://image.tmdb.org/t/p/w400/${movie.poster_path}`}
		alt={movie.title}
		class="h-100 w-full rounded-lg object-contain"
	/>
	<div class="overlay">
		<h2 class="text-lg font-semibold">{movie.title}</h2>
	</div>
</a>

<style>
	.movie-preview {
		position: relative;
		cursor: pointer;
		overflow: hidden;
	}

	.movie-preview img {
		width: 100%;
		object-fit: cover;
	}

	.movie-preview:hover {
		transform: scale(1.05);
	}

	.overlay {
		position: absolute;
		bottom: 0;
		left: 0;
		width: 100%;
		background: rgba(0, 0, 0, 0.7);
		color: white;
		padding: 15px;
		text-align: center;
		opacity: 0;
		transition: opacity 0.3s;
	}

	.movie-preview:hover .overlay {
		opacity: 1;
	}

</style>