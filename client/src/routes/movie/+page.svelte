<script lang="ts">
    import MoviePreview from '$lib/components/MoviePreview.svelte';
    import type {PageProps} from './$types';

    let {data}: PageProps = $props();
</script>

<main class="bg-gray-900 text-white p-4 pb-20">
    <div class="max-w-4xl mx-auto">
        <h1 class="mt-15 mb-10 text-4xl font-bold">Movie Recommendations</h1>
        <div class="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3">
            {#each data.movies as movie}
                <MoviePreview {movie}/>
            {/each}
        </div>
    </div>
</main>
