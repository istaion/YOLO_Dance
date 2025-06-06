# app/config.py
# MODEL_PATH = "model.pt" # A adapter
# CLASS_NAMES = [] # Si besoin
from datetime import datetime, timedelta, timezone
import jwt
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
import bcrypt
from typing import Dict
from fastapi import HTTPException, status, Depends
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Secret key and algorithm for JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# OAuth2 authentication scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """
    Generates a JWT access token.

    This function creates and encodes a JWT token with the specified payload
    and expiration time. If no expiration time is provided, the default expiration
    time will be used.

    Args:
        data (dict): The payload data to encode in the token.
        expires_delta (timedelta, optional): The expiration duration for the token.

    Returns:
        str: The encoded JWT token.
    """
    # Copy the provided data and add expiration time
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})  # Add expiration to the payload
    
    # Encode the payload into a JWT token using the secret key and algorithm
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Dict:
    """
    Decodes and verifies a JWT token.

    This function decodes the provided JWT token using the secret key and algorithm.
    If the token is expired or invalid, it raises an HTTPException with a relevant message.

    Args:
        token (str): The JWT token to decode and verify.

    Returns:
        Dict: The decoded token payload, if the token is valid.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        # Decode the token using the secret key and algorithm
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        # Raise an exception if the token is expired
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        # Raise an exception if the token is invalid
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(token: str = Depends(oauth2_scheme)) -> int:
    """
    Retrieves the current authenticated user id from the JWT token.

    This function decodes the provided JWT token, verifies its validity, and extracts
    the user information from the database. If the user is not found or the token is invalid,
    an HTTPException is raised.

    Args:
        token (str): The JWT token from the request header.
        db (Session): The database session used to query the user.

    Returns:
        User id: The authenticated user id.

    Raises:
        HTTPException: If the token is invalid, expired, or the user is not found in the database.
    """
    try:
        # Verify and decode the token, then extract user data from it
        user_data = verify_token(token)
        # Query the user from the database using the decoded email
        user_id = dict(get_user_by_username(user_data['sub']))["id"]
        
        if user_id is None:
            # Raise exception if the user is not found in the database
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except Exception:
        # Raise a general exception if there is any error with the token or user lookup
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)