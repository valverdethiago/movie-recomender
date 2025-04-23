import { fail, redirect } from '@sveltejs/kit';
import type { Actions } from './$types';
import { convertLoginToUserId } from '$lib/utils';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ cookies }) => {
	const authCookie = cookies.get('auth');

	if (!!authCookie) {
		return redirect(302, '/movie');
	}
};

export const actions: Actions = {
	default: async ({ request, cookies }) => {
		const formData = await request.formData();
		const login = formData.get('login');
		const password = formData.get('password');

		const userId = convertLoginToUserId(String(login));

		if (password == 'poc') {
			cookies.set('auth', String(userId), {
				path: '/',
				maxAge: 60 * 60 * 24 * 365,
				httpOnly: false
			});

			redirect(302, '/');
		}

		return fail(400, { error: 'Invalid login or password' });
	}
};
