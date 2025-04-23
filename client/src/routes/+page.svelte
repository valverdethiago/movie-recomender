<script lang="ts">
	import { enhance } from '$app/forms';
	export let form: { error?: string } | null;
</script>

<main class="flex min-h-screen items-center justify-center bg-gray-100">
	<form class="w-96 rounded-lg bg-white p-8 shadow-md" method="POST" use:enhance>
		<h2 class="mb-4 text-center text-2xl font-semibold">Sign in</h2>
		<div class="mb-4">
			<label for="login" class="block text-gray-700">Login:</label>
			<input
				type="login"
				id="login"
				name="login"
				class="w-full rounded-lg border border-gray-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
			/>
		</div>
		<div class="mb-4">
			<label for="password" class="block text-gray-700">Password:</label>
			<input
				type="password"
				id="password"
				name="password"
				class="w-full rounded-lg border border-gray-300 px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
			/>
		</div>
		{#if form?.error}
			<p class="mb-4 text-center text-red-500">{form?.error}</p>
		{/if}
		<button class="w-full rounded-lg bg-blue-500 py-2 text-white transition hover:bg-blue-600"
			>Entrar</button
		>
	</form>
</main>
