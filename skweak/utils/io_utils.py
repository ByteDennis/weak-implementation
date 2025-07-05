# utils/io_utils.py
import inspect
import shutil
import csv
import json
import pickle
import hashlib
import functools
from loguru import logger
from typing import List, Literal, Optional, Union, Any, Dict, Set, Callable
from upath import UPath
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from ..constant import TIME_FORMAT


class IOProcessor:
    """Data processing utilities for experiments."""

    @staticmethod
    def load_json(filepath: UPath) -> Dict[str, Any]:
        # Load JSON data from file
        with open(filepath, "r") as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Dict[str, Any], filepath: UPath) -> None:
        # Save data as JSON file
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_pickle(filepath: UPath) -> Any:
        # Load pickled object
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def save_pickle(obj: Any, filepath: UPath) -> None:
        # Save object using pickle
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_csv(filepath: UPath) -> List[Dict[str, Any]]:
        # Load CSV data as list of dictionaries
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def save_csv(data: List[Dict[str, Any]], filepath: UPath) -> None:
        # Save data as CSV file
        if not data:
            return

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


class FileManager:
    """File operations utility class for experiment framework."""

    @staticmethod
    def ensure_dir(path: UPath) -> UPath:
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def copy_files(
        source: UPath, destination: UPath, pattern: str = "*"
    ) -> List[UPath]:
        # Copy files matching pattern from source to destination
        FileManager.ensure_dir(destination)
        copied_files = []

        for file_path in source.glob(pattern):
            if file_path.is_file():
                dest_path = destination / file_path.name
                shutil.copy2(file_path, dest_path)
                copied_files.append(dest_path)

        return copied_files

    @staticmethod
    def safe_remove(path: UPath) -> bool:
        # Safely remove file or directory
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            return True
        except Exception as e:
            logger.error(f"Failed to remove {path}: {str(e)}")
            return False


@dataclass
class CacheMeta:
    """
    Metadata container for cached function results tracking.

    This dataclass stores comprehensive information about cached function calls
    including timing, expiration, usage statistics, and file system details.
    It enables intelligent cache management with expiration handling and
    usage analytics for performance optimization.
    """

    function_name: str
    first_arg: Optional[str]
    force_restart: bool
    last_call: str
    expire_hours: Optional[float]
    expire_time: Optional[str]
    hit_count: int
    file_path: str
    file_size: str


class CacheUtils:
    """
    Advanced function result caching utility with expiration and metadata tracking.
    """

    def __init__(
        self,
        expire_hours: Optional[float] = None,
        directory: str = "./cache",
        cache_name: Optional[str] = None,
        ignore_args: Optional[str | Set[str]] = None,
        force_restart: bool = False,
        verbose: bool = False,
    ):
        self.expire_hours = expire_hours
        self.directory = UPath(directory)
        self.name = cache_name
        self.ignore_args = self._to_set(ignore_args)
        self.force_restart = force_restart
        self.verbose = verbose
        self.meta_path = self.directory / "meta.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta: Dict[str, CacheMeta] = self._load_meta()
        self.time_format = TIME_FORMAT

    def _load_meta(self) -> Dict[str, CacheMeta]:
        # Load metadata from disk or return empty dict
        try:
            d = load_file(self.meta_path)
            return {k: CacheMeta(**v) for k, v in d.items()}
        except Exception as e:
            return {}

    def _save_meta(self):
        d = {k: asdict(v) for k, v in self.meta.items()}
        save_file(d, self.meta_path)

    def _to_set(self, items: Optional[str | Set[str]]) -> Set[str]:
        # Convert string or set to set
        return {items} if isinstance(items, str) else set(items or [])

    def _log(self, msg: str):
        # Log message if verbose mode enabled
        if self.verbose:
            logger.info(msg)

    def _get_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        # Generate unique cache key from function signature and arguments
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        filtered = {
            k: v for k, v in bound.arguments.items() if k not in self.ignore_args
        }
        key_data = f"{func.__name__}_{sorted(filtered.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_filepath(self, func: Callable, cache_key: str) -> UPath:
        # Generate file path for cached result
        fname = self.name or f"{func.__name__}_{cache_key}.pkl"
        return self.directory / fname

    def _get_file_size_str(self, path: UPath) -> str:
        try:
            size = path.stat().st_size
            for unit in ["B", "KB", "MB", "GB"]:
                if size < 1024:
                    return f"{size:.2f} {unit}"
                size /= 1024
            return f"{size:.2f} TB"
        except:
            return "Unknown"

    def _is_expired(self, meta: CacheMeta) -> bool:
        if self.expire_hours:
            elapsed = (
                datetime.now() - datetime.strptime(meta.last_call, self.time_format)
            ).seconds / 3600
            return elapsed > self.expire_hours
        # Check if cache entry has expired based on metadata
        if not meta.expire_hours or meta.expire_time == "Never":
            return False
        try:
            current_time = datetime.now().strftime(self.time_format)
            return current_time > meta.expire_time
        except Exception as e:
            self._log(f"[Expiration Check Error] {e}")
            return True  # Treat parsing errors as expired

    def _is_valid(self, path: UPath, meta: CacheMeta) -> bool:
        # Check if cache file exists and hasn't expired
        if not path.exists():
            return False
        if self._is_expired(meta):
            self._log(f"Cache expired: {path}")
            self._remove_expired_cache(path, meta)
            return False
        return True

    def _remove_expired_cache(self, path: UPath, meta: CacheMeta):
        # Remove expired cache file and update metadata
        try:
            if path.exists():
                path.unlink()
                self._log(f"Removed expired cache: {path}")
            # Remove from metadata
            cache_key = None
            for key, cached_meta in self.meta.items():
                if cached_meta.file_path == str(path):
                    cache_key = key
                    break
            if cache_key:
                del self.meta[cache_key]
                self._save_meta()
        except Exception as e:
            self._log(f"[Cache Removal Error] {e}")

    def _update_meta(self, key: str, func: Callable, args: tuple, path: UPath):
        # Update cache metadata with current call information
        now = datetime.now()
        expire_time = "Never"

        if self.expire_hours:
            expire_instant = now + timedelta(hours=self.expire_hours)
            expire_time = expire_instant.strftime(self.time_format)

        first_arg = str(args[0]) if args else None
        size = self._get_file_size_str(path)

        now_str = now.strftime(self.time_format)
        if key in self.meta and not self.force_restart:
            self.meta[key].hit_count += 1
            self.meta[key].last_call = now_str
        else:
            self.meta[key] = CacheMeta(
                function_name=func.__name__,
                first_arg=first_arg,
                force_restart=self.force_restart,
                last_call=now_str,
                expire_hours=self.expire_hours,
                expire_time=expire_time,
                hit_count=0,
                file_path=str(path),
                file_size=size,
            )
        self._save_meta()

    def _cleanup_expired_caches(self):
        # Remove all expired cache entries
        expired_keys = []
        for key, meta in self.meta.items():
            if self._is_expired(meta):
                cache_path = UPath(meta.file_path)
                if cache_path.exists():
                    cache_path.unlink()
                    self._log(f"Cleaned up expired cache: {cache_path}")
                expired_keys.append(key)
        # Remove expired entries from metadata
        for key in expired_keys:
            del self.meta[key]
        if expired_keys:
            self._save_meta()

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # self._cleanup_expired_caches()

            force_restart = kwargs.pop("force_restart", self.force_restart)
            key = self._get_cache_key(func, args, kwargs)
            path = self._get_filepath(func, key)

            # Check if we have valid cached result
            if key in self.meta and not force_restart:
                meta = self.meta[key]

                # Check if cache is valid and not expired
                if self._is_valid(path, meta):
                    result = load_file(path)
                    if result is not None:
                        self._log(f"Cache hit: {path}")
                        self._update_meta(key, func, args, path)
                        return result

            # Cache miss - execute function and cache result
            self._log(f"Cache miss: {func.__name__}()")
            result = func(*args, **kwargs)
            save_file(result, path)
            self._update_meta(key, func, args, path)
            return result

        def cache_info(*args, scope: Literal["func", "all"] = "func", **kwargs):
            try:
                key = self._get_cache_key(func, args, kwargs)
                if key in self.meta:
                    return self.meta[key]
            except TypeError:
                pass
            match scope:
                case "func":
                    return {
                        k: v
                        for k, v in self.meta.items()
                        if v.function_name == func.__name__
                    }
                case "all":
                    return self.meta

        wrapper.clear_cache = lambda: self._clear_cache(func)
        wrapper.cache_info = cache_info
        return wrapper

    def _clear_cache(self, func: Callable):
        # Clear all cache files for specific function
        pattern = self.name or f"{func.__name__}_"
        count = 0

        for file_path in self.directory.iterdir():
            if file_path.name.startswith(pattern) and file_path.suffix == ".pkl":
                file_path.unlink()
                count += 1

        # Remove corresponding metadata entries
        keys_to_remove = []
        for key, meta in self.meta.items():
            if meta.function_name == func.__name__:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.meta[key]

        if keys_to_remove:
            self._save_meta()

        self._log(f"Cleared {count} cache file(s) for {func.__name__}")


def cache(
    expire_hours: Optional[float] = None,
    directory: str = ".cache",
    name: Optional[str] = None,
    ignore_args=None,
    force_restart: bool = False,
    verbose: bool = False,
):
    """Convenience function to create a cache decorator."""
    return CacheUtils(
        expire_hours=expire_hours,
        directory=directory,
        cache_name=name,
        ignore_args=ignore_args,
        force_restart=force_restart,
        verbose=verbose,
    )


def ensure_dir(path: str | UPath) -> UPath:
    # Standalone function to ensure directory exists
    path = UPath(path)
    return FileManager.ensure_dir(path)


def copy_file(
    source: str | UPath, dest: str | UPath, pattern: str = "*"
) -> List[UPath]:
    # Standalone function to copy files
    return FileManager.copy_files(UPath(source), UPath(dest), pattern)


def load_file(filepath: str | UPath, format_type: Optional[str] = None) -> Any:
    # Auto-detect and load data from various formats
    filepath = UPath(filepath)

    if format_type is None:
        format_type = filepath.suffix.lower()

    if format_type in [".json", "json"]:
        return IOProcessor.load_json(filepath)
    elif format_type in [".pkl", ".pickle", "pickle"]:
        return IOProcessor.load_pickle(filepath)
    elif format_type in [".csv", "csv"]:
        return IOProcessor.load_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def save_file(
    data: Any, filepath: str | UPath, format_type: Optional[str] = None
) -> None:
    # Auto-detect and save data in various formats
    filepath = UPath(filepath)

    if format_type is None:
        format_type = filepath.suffix.lower()

    if format_type in [".json", "json"]:
        IOProcessor.save_json(data, filepath)
    elif format_type in [".pkl", ".pickle", "pickle"]:
        IOProcessor.save_pickle(data, filepath)
    elif format_type in [".csv", "csv"]:
        IOProcessor.save_csv(data, filepath)
    else:
        raise ValueError(f"Unsupported format: {format_type}")


__all__ = ["ensure_dir", "copy_file", "load_file", "save_file", "cache"]
