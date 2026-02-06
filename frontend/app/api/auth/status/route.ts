import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
const SESSION_COOKIE_NAME = 'bench_session';

export async function GET(request: NextRequest) {
  const sessionId = request.cookies.get(SESSION_COOKIE_NAME)?.value;

  try {
    // Call backend with session cookie forwarded
    const response = await fetch(`${BACKEND_URL}/api/auth/status`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'Cookie': sessionId ? `${SESSION_COOKIE_NAME}=${sessionId}` : '',
      },
    });

    const data = await response.json();

    const next = NextResponse.json(data);
    // Refresh cookie expiry so users stay signed in for 30 days after last usage.
    if (sessionId && data?.authenticated) {
      next.cookies.set({
        name: SESSION_COOKIE_NAME,
        value: sessionId,
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        path: '/',
        maxAge: 60 * 60 * 24 * 30,
      });
    }

    return next;
  } catch (error) {
    console.error('Error proxying auth status:', error);
    return NextResponse.json({
      idp_configured: false,
      authenticated: false,
      user: null,
      has_invoke_scope: false,
    });
  }
}





























