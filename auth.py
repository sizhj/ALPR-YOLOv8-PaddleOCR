"""
auth.py
JWT 用户认证模块

提供管理员登录接口，返回 JWT token。
后续受保护接口通过 get_current_user 依赖注入验证身份。

依赖：
    pip install python-jose[cryptography] passlib[bcrypt]
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from models import User, get_db, SessionLocal
from schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["认证"])

# ── 配置 ──────────────────────────────────────────────────
SECRET_KEY      = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM       = "HS256"
TOKEN_EXPIRE_HOURS = 24

pwd_context     = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme   = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── 工具函数 ─────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {"sub": username, "exp": expire},
        SECRET_KEY, algorithm=ALGORITHM
    )


# ── 依赖注入：获取当前登录用户 ────────────────────────────
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    从 JWT token 中解析用户，用于保护需要登录的接口。
    使用方式：在接口参数中加 current_user: User = Depends(get_current_user)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token 无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user


# ── 接口 ─────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse, summary="管理员登录")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    账号密码登录，返回 JWT token。
    后续请求在 Header 中携带：Authorization: Bearer <token>
    """
    user = db.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )
    return TokenResponse(access_token=create_token(user.username))


@router.get("/me", summary="获取当前登录用户信息")
def get_me(current_user: User = Depends(get_current_user)):
    return {
        "id":       current_user.id,
        "username": current_user.username,
        "is_active": current_user.is_active
    }


# ── 初始化默认管理员（首次运行） ──────────────────────────
def init_admin():
    """创建默认管理员账号 admin/admin123，仅首次运行时执行"""
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                username="admin",
                hashed_password=hash_password("admin123")
            )
            db.add(admin)
            db.commit()
            print("默认管理员已创建：admin / admin123")
    finally:
        db.close()
