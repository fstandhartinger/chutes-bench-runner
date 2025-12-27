import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
const SESSION_COOKIE_NAME = 'bench_session';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get('code');
  const state = searchParams.get('state');
  const error = searchParams.get('error');
  const errorDescription = searchParams.get('error_description');

  // Handle OAuth errors
  if (error) {
    const errorUrl = new URL('/auth/error', request.url);
    errorUrl.searchParams.set('error', error);
    if (errorDescription) {
      errorUrl.searchParams.set('description', errorDescription);
    }
    return NextResponse.redirect(errorUrl);
  }

  if (!code || !state) {
    return NextResponse.redirect(new URL('/auth/error?error=missing_params', request.url));
  }

  try {
    // Exchange code for session via backend
    const callbackUrl = new URL('/api/auth/callback', BACKEND_URL);
    callbackUrl.searchParams.set('code', code);
    callbackUrl.searchParams.set('state', state);

    const response = await fetch(callbackUrl.toString(), {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      const errorUrl = new URL('/auth/error', request.url);
      errorUrl.searchParams.set('error', 'token_exchange_failed');
      errorUrl.searchParams.set('description', errorData.detail || 'Failed to exchange code');
      return NextResponse.redirect(errorUrl);
    }

    const data = await response.json();
    
    if (!data.success || !data.session_id) {
      return NextResponse.redirect(new URL('/auth/error?error=no_session', request.url));
    }

    // Create response with redirect to return_to or home
    const returnTo = data.return_to || '/';
    const redirectResponse = NextResponse.redirect(new URL(returnTo, request.url));

    // Set session cookie (httpOnly, secure in production)
    redirectResponse.cookies.set({
      name: SESSION_COOKIE_NAME,
      value: data.session_id,
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 30, // 30 days
    });

    return redirectResponse;
  } catch (error) {
    console.error('OAuth callback error:', error);
    return NextResponse.redirect(new URL('/auth/error?error=server_error', request.url));
  }
}

