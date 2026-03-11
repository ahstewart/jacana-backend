from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import Session
from supabase import create_client, Client
from typing import Optional

from database import get_session
from schema import UserDB
from config import get_settings

settings = get_settings()

# Initialize the Supabase client using your URL and the public ANON API key
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

# ==========================================
# THE BOUNCER
# ==========================================
def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security), 
    session: Session = Depends(get_session)
) -> UserDB:
    
    token = creds.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # 1. Online Validation: Ping Supabase Auth API
        # This completely replaces the old JWT math and JWKS fetching!
        user_response = supabase.auth.get_user(token)
        
        if not user_response or not user_response.user:
            raise credentials_exception
            
        sb_user = user_response.user
        user_id = sb_user.id
        email = sb_user.email
        
    except Exception as e:
        print(f"[AUTH ERROR] Token verification failed: {e}")
        raise credentials_exception

    # 2. Database Lookup
    user = session.get(UserDB, user_id)

    # 3. Just-in-Time (JIT) User Provisioning
    if not user:
        user = UserDB(
            id=user_id,
            email=email,
            username=email.split("@")[0] if email else "unknown",
            is_developer=False,
            created_at=sb_user.created_at # Supabase returns this as a string automatically
        )
        session.add(user)
        session.commit()
        session.refresh(user)

    return user


def get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
    session: Session = Depends(get_session),
) -> Optional[UserDB]:
    """Like get_current_user but returns None instead of 401 when unauthenticated."""
    if not creds:
        return None
    try:
        user_response = supabase.auth.get_user(creds.credentials)
        if not user_response or not user_response.user:
            return None
        user = session.get(UserDB, user_response.user.id)
        return user
    except Exception:
        return None