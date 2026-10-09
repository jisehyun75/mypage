"""읽어 들인 CBF·복수예가 표를 사용자 폴더에 캐시한다(같은 파일이면 두 번째부터 몇 초).

캐시 이름에는 원본 경로, 파일 크기·수정시각, 이 패키지의 코드 전체와 pandas 버전이 들어간다.
원본 파일을 고치거나 로더가 바뀌면 자동으로 다시 읽는다. 캐시를 읽지 못하면 조용히 다시 만든다.
캐시는 이 프로그램이 직접 만든 파일만 읽는다(기본 위치: 사용자 폴더의 .nara_bid_stat_cache).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

def default_cache_dir() -> Path:
    return Path(os.environ.get("NARA_CACHE_DIR") or (Path.home() / ".nara_bid_stat_cache"))


def _code_hash() -> str:
    """패키지의 모든 .py 와 pandas 버전(코드가 바뀌면 캐시를 다시 만든다)."""
    h = hashlib.sha1(pd.__version__.encode())
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _signature(sources: Iterable[Path]) -> str:
    h = hashlib.sha1(_code_hash().encode())
    for p in sorted(Path(s).resolve() for s in sources):
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:20]


def cached_frame(kind: str, origin: str, sources: Iterable[str | Path], build: Callable[[], pd.DataFrame],
                 *, cache_dir: str | Path | None = None, log: Callable[[str], None] | None = None) -> pd.DataFrame:
    """sources(원본 파일들)가 그대로면 캐시를, 아니면 build() 결과를 저장해 돌려준다.

    origin: 원본을 구분하는 문자열(예: 절대경로|시트). 같은 origin 의 예전 캐시는 새로 저장할 때 지운다.
    """
    sources = [Path(s) for s in sources]
    d = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    origin_key = hashlib.sha1(str(origin).encode()).hexdigest()[:10]
    try:
        sig = _signature(sources)
    except OSError:
        return build()
    path = d / f"{kind}_{origin_key}_{sig}.pkl"
    if path.exists():
        try:
            df = pd.read_pickle(path)
            if log:
                log(f"[cache] {kind}: 저장된 표 사용({path.name})")
            return df
        except Exception:  # 손상·버전 차이 -> 다시 만든다
            pass
    df = build()
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        df.to_pickle(tmp)
        os.replace(tmp, path)
        for old in d.glob(f"{kind}_{origin_key}_*.pkl"):  # 같은 원본의 예전 캐시 정리
            if old != path:
                old.unlink(missing_ok=True)
    except OSError:
        pass  # 캐시를 못 써도 분석은 계속
    return df
