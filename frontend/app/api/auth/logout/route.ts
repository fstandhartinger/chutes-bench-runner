import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
const SESSION_COOKIE_NAME = 'bench_session';

export async function POST(request: NextRequest) {
  const sessionId = request.cookies.get(SESSION_COOKIE_NAME)?.value;

  try {
    // Call backend to revoke tokens
    if (sessionId) {
      await fetch(`${BACKEND_URL}/api/auth/logout`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Cookie': `${SESSION_COOKIE_NAME}=${sessionId}`,
        },
      });
    }
  } catch (error) {
    console.error('Error calling backend logout:', error);
  }

  // Clear the session cookie
  const response = NextResponse.json({ success: true });
  response.cookies.set({
    name: SESSION_COOKIE_NAME,
    value: '',
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: 0,
  });

  return response;
}
















