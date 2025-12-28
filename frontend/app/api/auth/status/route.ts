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
    
    return NextResponse.json(data);
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




