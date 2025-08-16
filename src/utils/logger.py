"""로깅 설정 모듈"""

import logging
import os
import sys
from typing import Optional


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """모듈별 로거 설정"""
    logger = logging.getLogger(name)

    # 이미 핸들러가 설정되어 있으면 중복 설정 방지
    if logger.handlers:
        return logger

    # 로그 레벨 설정 (환경변수 또는 기본값 INFO)
    log_level = level or "INFO"
    logger.setLevel(getattr(logging, log_level.upper()))

    # 콘솔 핸들러 생성
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level.upper()))

    # 포맷터 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # 상위 로거로의 전파 방지

    return logger


def set_global_log_level(level: str):
    """전체 로그 레벨 설정"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(log_level)

    # 기존 핸들러들의 레벨도 업데이트
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)
