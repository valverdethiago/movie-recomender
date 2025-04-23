<script lang="ts">
    import type {PageProps} from './$types';
    import {ArrowLeft} from 'lucide-svelte';
    import {goto} from '$app/navigation';

    let {data}: PageProps = $props();

    function goBack() {
        goto('/movie');
    }
</script>

{#if data.movie}
    <main class="bg-gray-900 text-white pb-16 min-h-screen">
        <div class="inset-0 z-0">
            <img
                src={`https://image.tmdb.org/t/p/original/${data.movie.backdrop_path}`}
                alt="Backdrop"
                class="w-full h-80 object-cover opacity-50"
            />
        </div>
        <div class="max-w-4xl mx-auto px-5 container">
            <div class="flex flex-col md:flex-row mt-10">
                <div class="top-0 relative mt-10 md:absolute">
                    <button class="flex items-center gap-2 bg-transparent border-none text-white cursor-pointer text-base mb-5 transition-opacity duration-200 hover:opacity-75"
                            on:click={goBack}>
                        <ArrowLeft size={24}/>
                        <span>Back</span>
                    </button>

                    <img
                        src={`https://image.tmdb.org/t/p/w400/${data.movie.poster_path}`}
                        alt={data.movie.title}
                        class="rounded-lg shadow-lg transition-transform duration-300 ease-in-out hover:scale-105 mt-20 mb-10 w-full md:w-auto"
                    />
                </div>

                <div class="flex-1/2"></div>

                <div class="flex flex-col items-start md:flex-row gap-5 ml-auto flex-1/2">
                    <div class="w-full md:ml-4 flex-1/2">
                        <p class="flex justify-between text-sm text-gray-400 mb-2">
                            <span>{data.movie.genres.join(', ')}</span>
                            <span>{data.movie.year}</span>
                        </p>
                        <h1 class="text-3xl font-bold">{data.movie.original_title}</h1>
                        <p class="mt-2 text-gray-500">{data.movie.description}</p>
                    </div>
                </div>
            </div>
        </div>
    </main>
{:else}
    <div class="flex items-center justify-center min-h-screen bg-gray-900">
        <div class="flex flex-col items-center">
            <div class="loader ease-linear rounded-full border-8 border-t-8 border-gray-200 h-32 w-32 mb-4"></div>
            <p class="text-white text-xl font-semibold">Loading...</p>
        </div>
    </div>
{/if}

<style>
    .loader {
        -webkit-animation: spin 1s linear infinite;
        animation: spin 1s linear infinite;
    }

    @-webkit-keyframes spin {
        0% {
            -webkit-transform: rotate(0deg);
        }
        100% {
            -webkit-transform: rotate(360deg);
        }
    }

    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }
        100% {
            transform: rotate(360deg);
        }
    }

    @media (max-width: 800px) {
        .container {
            position: relative;
            top: -350px;
        }
    }
</style>