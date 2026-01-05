from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.database.connection import mongo_db_dependency
from app.repositories.user_repository import UserRepository
from app.schemas.user import (
    Token, 
    UserCreate, 
    UserPublic, 
    LoginRequest, 
    RefreshTokenRequest
)
from app.services.user_service import UserService
from app.utils.security import (
    create_access_token, 
    create_refresh_token, 
    decode_refresh_token
)
from app.utils.dependencies import get_current_user


router = APIRouter(prefix="/auth", tags=["auth"])


# Khởi tạo UserService ở đây để tái sử dụng
async def get_user_service(db = Depends(mongo_db_dependency)) -> UserService:
    """Dependency inject UserService với UserRepository"""
    user_repo = UserRepository(db)
    return UserService(user_repo)


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserCreate, user_service: UserService = Depends(get_user_service)) -> UserPublic:
    """
    Đăng ký tài khoản mới.
    
    - **email**: Email hợp lệ (unique)
    - **password**: Mật khẩu (tối thiểu 6 ký tự)
    - **full_name**: Tên đầy đủ (tùy chọn)
    """
    try:
        user = await user_service.register_user(
            email=payload.email,
            password=payload.password,
            full_name=payload.full_name
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=Token)
async def login(
    payload: LoginRequest, 
    user_service: UserService = Depends(get_user_service)
) -> Token:
    """
    Đăng nhập với email và password (JSON body).
    
    Trả về:
    - **access_token**: Token để gọi API (hết hạn sau 60 phút)
    - **refresh_token**: Token để làm mới access_token (hết hạn sau 7 ngày)
    - **user**: Thông tin user
    """
    # Xác thực user qua Service
    user = await user_service.authenticate_user(payload.email, payload.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Email hoặc mật khẩu không đúng"
        )
    
    # Tạo access token và refresh token
    user_id = user["_id"]
    access_token = create_access_token(subject=user_id)
    refresh_token = create_refresh_token(subject=user_id)
    
    # Tạo user public response
    user_public = UserPublic(
        id=user_id,
        email=user["email"],
        full_name=user.get("full_name"),
        role=user.get("role", "user")
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user_public
    )


@router.post("/login/form", response_model=Token)
async def login_form(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    user_service: UserService = Depends(get_user_service)
) -> Token:
    """
    Đăng nhập với form-data (OAuth2 standard).
    Dùng cho Swagger UI testing.
    """
    user = await user_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Email hoặc mật khẩu không đúng"
        )
    
    user_id = user["_id"]
    access_token = create_access_token(subject=user_id)
    refresh_token = create_refresh_token(subject=user_id)
    
    user_public = UserPublic(
        id=user_id,
        email=user["email"],
        full_name=user.get("full_name"),
        role=user.get("role", "user")
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user_public
    )


@router.get("/me", response_model=UserPublic)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user)
) -> UserPublic:
    """
    Lấy thông tin user hiện tại từ access token.
    
    Yêu cầu: Header `Authorization: Bearer <access_token>`
    """
    return UserPublic(
        id=current_user["_id"],
        email=current_user["email"],
        full_name=current_user.get("full_name"),
        role=current_user.get("role", "user")
    )


@router.post("/refresh", response_model=Token)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    db = Depends(mongo_db_dependency)
) -> Token:
    """
    Làm mới access token bằng refresh token.
    
    Khi access token hết hạn (401), client gọi endpoint này
    với refresh_token để lấy cặp token mới.
    """
    try:
        # Decode refresh token
        token_payload = decode_refresh_token(payload.refresh_token)
        user_id = token_payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Verify user still exists
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Tạo token mới
        new_access_token = create_access_token(subject=user_id)
        new_refresh_token = create_refresh_token(subject=user_id)
        
        user_public = UserPublic(
            id=user_id,
            email=user["email"],
            full_name=user.get("full_name"),
            role=user.get("role", "user")
        )
        
        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            user=user_public
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Đăng xuất (client-side).
    
    Backend không lưu trạng thái session, nên logout chỉ là
    xác nhận request hợp lệ. Client tự xóa tokens khỏi localStorage.
    """
    return {"message": "Logged out successfully", "user_id": current_user["_id"]}


@router.post("/seed-test-user", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def seed_test_user(user_service: UserService = Depends(get_user_service)) -> UserPublic:
    """
    Tạo test user để development.
    
    Credentials:
    - Email: test@example.com
    - Password: secret123
    """
    user = await user_service.get_or_create_test_user(
        email="test@example.com",
        password="secret123",
        full_name="Test User",
        role="user"
    )
    return user


@router.post("/seed-admin", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
async def seed_admin_user(user_service: UserService = Depends(get_user_service)) -> UserPublic:
    """
    Tạo admin user để development.
    
    Credentials:
    - Email: admin@example.com
    - Password: admin123
    """
    user = await user_service.get_or_create_test_user(
        email="admin@example.com",
        password="admin123",
        full_name="Admin User",
        role="admin"
    )
    return user
