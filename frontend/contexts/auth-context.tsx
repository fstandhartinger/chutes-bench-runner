'use client';

import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

interface User {
  id: string;
  username: string;
}

interface AuthState {
  isLoading: boolean;
  isAuthenticated: boolean;
  idpConfigured: boolean;
  user: User | null;
  hasInvokeScope: boolean;
}

interface AuthContextType extends AuthState {
  login: (returnTo?: string) => void;
  logout: () => Promise<void>;
  refreshAuthState: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    isLoading: true,
    isAuthenticated: false,
    idpConfigured: false,
    user: null,
    hasInvokeScope: false,
  });

  const refreshAuthState = useCallback(async () => {
    try {
      // Use frontend API route that proxies to backend with cookie
      const response = await fetch('/api/auth/status', {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        setState({
          isLoading: false,
          isAuthenticated: data.authenticated,
          idpConfigured: data.idp_configured,
          user: data.user,
          hasInvokeScope: data.has_invoke_scope,
        });
      } else {
        setState(prev => ({
          ...prev,
          isLoading: false,
          isAuthenticated: false,
          user: null,
        }));
      }
    } catch (error) {
      console.error('Failed to fetch auth status:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
      }));
    }
  }, []);

  useEffect(() => {
    refreshAuthState();
  }, [refreshAuthState]);

  const login = useCallback((returnTo?: string) => {
    const loginUrl = new URL('/api/auth/login', BACKEND_URL);
    if (returnTo) {
      loginUrl.searchParams.set('return_to', returnTo);
    }
    window.location.href = loginUrl.toString();
  }, []);

  const logout = useCallback(async () => {
    try {
      // Use frontend API route that handles cookie deletion
      await fetch('/api/auth/logout', {
        method: 'POST',
        credentials: 'include',
      });
      
      // Clear client-side state
      setState({
        isLoading: false,
        isAuthenticated: false,
        idpConfigured: state.idpConfigured,
        user: null,
        hasInvokeScope: false,
      });
      
      // Redirect to home
      window.location.href = '/';
    } catch (error) {
      console.error('Logout failed:', error);
    }
  }, [state.idpConfigured]);

  return (
    <AuthContext.Provider
      value={{
        ...state,
        login,
        logout,
        refreshAuthState,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

