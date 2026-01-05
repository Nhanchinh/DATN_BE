from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field


UserRole = Literal["admin", "user"]


class UserBase(BaseModel):

    email: EmailStr


class UserCreate(UserBase):

    password: str = Field(min_length=6)
    full_name: Optional[str] = None


class UserInDB(UserBase):

    id: str
    full_name: Optional[str] = None
    role: UserRole = "user"


class UserPublic(UserBase):

    id: str
    full_name: Optional[str] = None
    role: UserRole = "user"


class Token(BaseModel):
    """Token response with access_token, refresh_token and user info"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserPublic"


class TokenPayload(BaseModel):

    sub: str
    exp: int


class LoginRequest(BaseModel):
    """JSON login request body"""
    email: EmailStr
    password: str = Field(min_length=6)


class RefreshTokenRequest(BaseModel):
    """Request body for token refresh"""
    refresh_token: str
