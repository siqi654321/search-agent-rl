import os
import socket
import subprocess
import threading
import time
from typing import Optional

import ray
import requests


def _build_probe_urls(port: int) -> list[str]:
    urls = [f"http://127.0.0.1:{port}"]
    host = ray.util.get_node_ip_address().strip("[]")
    if host and host not in {"127.0.0.1", "localhost"}:
        if ":" in host:
            urls.append(f"http://[{host}]:{port}")
        else:
            urls.append(f"http://{host}:{port}")
    return urls


def _build_public_url(port: int) -> str:
    host = ray.util.get_node_ip_address().strip("[]")
    if ":" in host:
        return f"http://[{host}]:{port}"
    return f"http://{host}:{port}"


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_http_ready(urls: list[str], health_paths: list[str], timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        any_success = False
        for url in urls:
            for path in health_paths:
                try:
                    resp = requests.get(f"{url}{path}", timeout=2)
                    if resp.ok:
                        return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
                any_success = True
        if any_success:
            break
        time.sleep(1)
    raise RuntimeError(f"Service at {urls} is not ready after {timeout_s}s. last_error={last_error}")


def _probe_sglang_chat(base_urls: list[str], model: str, timeout_s: int = 20) -> tuple[str, list[str]]:
    errors = []
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    for base_url in base_urls:
        url = base_url.rstrip("/") + "/v1/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if resp.ok:
                return base_url, errors
            errors.append(f"{url} -> status={resp.status_code}, body={resp.text[:300]!r}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors) if errors else "unknown chat probe error")


def _wait_retrieval_ready(urls: list[str], readiness_paths: list[str], timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        any_success = False
        for url in urls:
            for path in readiness_paths:
                try:
                    resp = requests.get(f"{url}{path}", timeout=2)
                    if resp.status_code in (200, 405):
                        return
                    last_error = RuntimeError(f"unexpected status {resp.status_code} for {url}{path}")
                    any_success = True
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    continue
        if any_success:
            break
        time.sleep(1)
    raise RuntimeError(f"Retrieval service at {urls} is not ready after {timeout_s}s. last_error={last_error}")


class _BaseServiceActor:
    def _build_env(self) -> dict:
        env = os.environ.copy()
        gpu_ids = ray.get_gpu_ids()
        if gpu_ids:
            visible = ",".join(str(int(g)) if float(g).is_integer() else str(g) for g in gpu_ids)
            env["CUDA_VISIBLE_DEVICES"] = visible
        return env

    def _start_log_threads(self, name: str) -> None:
        if self.proc is None:
            return
        self._stdout_lines = []
        self._stderr_lines = []

        def _pump(pipe, sink, label):
            try:
                for line in iter(pipe.readline, ""):
                    if not line:
                        break
                    line = line.rstrip()
                    sink.append(line)
                    if len(sink) > 200:
                        del sink[:-200]
                    print(f"[{name}][{label}] {line}", flush=True)
            finally:
                try:
                    pipe.close()
                except Exception:
                    pass

        self._stdout_thread = threading.Thread(
            target=_pump, args=(self.proc.stdout, self._stdout_lines, "stdout"), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=_pump, args=(self.proc.stderr, self._stderr_lines, "stderr"), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _recent_logs(self) -> str:
        stdout_lines = getattr(self, "_stdout_lines", [])
        stderr_lines = getattr(self, "_stderr_lines", [])
        chunks = []
        if stdout_lines:
            chunks.append("stdout:\n" + "\n".join(stdout_lines[-50:]))
        if stderr_lines:
            chunks.append("stderr:\n" + "\n".join(stderr_lines[-50:]))
        return "\n\n".join(chunks) if chunks else "<no captured logs>"

    def get_visible_devices(self) -> str:
        gpu_ids = ray.get_gpu_ids()
        return ",".join(str(int(g)) if float(g).is_integer() else str(g) for g in gpu_ids)


@ray.remote(max_restarts=1, max_task_retries=0)
class SummarySGLangActor(_BaseServiceActor):
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        mem_fraction_static: float = 0.5,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        extra_args: Optional[list[str]] = None,
        startup_timeout_s: int = 300,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.mem_fraction_static = mem_fraction_static
        self.host = host
        self.port = port or _get_free_port()
        self.extra_args = extra_args or []
        self.startup_timeout_s = startup_timeout_s
        self.proc = None
        self.url = None

    def start(self) -> str:
        if self.proc is not None:
            return self.get_url()

        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        env = self._build_env()
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_log_threads("summary-sglang")
        probe_urls = _build_probe_urls(self.port)
        self.url = _build_public_url(self.port)
        try:
            _wait_http_ready(probe_urls, ["/health", "/health_generate", "/model_info"], self.startup_timeout_s)
            preferred_url, probe_errors = _probe_sglang_chat(probe_urls, model="default")
            self.url = preferred_url
            if probe_errors:
                print(f"[summary-sglang][probe] chat probe fallback to {preferred_url}; prior_errors={probe_errors}", flush=True)
            else:
                print(f"[summary-sglang][probe] chat probe ok via {preferred_url}", flush=True)
        except Exception as exc:
            raise RuntimeError(
                f"Summary service failed to become ready on {probe_urls}. cmd={cmd} recent_logs=\n{self._recent_logs()}"
            ) from exc
        return self.url

    def get_url(self) -> str:
        if self.url is None:
            self.url = _build_public_url(self.port)
        return self.url

    def get_pid(self) -> Optional[int]:
        return None if self.proc is None else self.proc.pid

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            self.proc = None


@ray.remote(max_restarts=1, max_task_retries=0)
class RetrievalServiceActor(_BaseServiceActor):
    def __init__(
        self,
        script_path: str,
        index_path: str,
        corpus_path: str,
        retriever_name: str,
        retriever_model: str,
        sglang_base_url: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        faiss_gpu: bool = True,
        extra_args: Optional[list[str]] = None,
        startup_timeout_s: int = 300,
    ):
        self.script_path = script_path
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.retriever_name = retriever_name
        self.retriever_model = retriever_model
        self.sglang_base_url = sglang_base_url
        self.host = host
        self.port = port or _get_free_port()
        self.faiss_gpu = faiss_gpu
        self.extra_args = extra_args or []
        self.startup_timeout_s = startup_timeout_s
        self.proc = None
        self.url = None

    def start(self) -> str:
        if self.proc is not None:
            return self.get_url()

        cmd = [
            "python3",
            self.script_path,
            "--index_path",
            self.index_path,
            "--corpus_path",
            self.corpus_path,
            "--retriever_name",
            self.retriever_name,
            "--retriever_model",
            self.retriever_model,
            "--sglang_base_url",
            self.sglang_base_url,
            "--host",
            self.host,
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        if self.faiss_gpu:
            cmd.insert(6, "--faiss_gpu")

        env = self._build_env()
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_log_threads("retrieval-service")
        probe_urls = _build_probe_urls(self.port)
        self.url = _build_public_url(self.port)
        try:
            _wait_retrieval_ready(
                probe_urls,
                ["/retrieve_summarize_compat", "/retrieve"],
                self.startup_timeout_s,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Retrieval service failed to become ready on {probe_urls}. cmd={cmd} recent_logs=\n{self._recent_logs()}"
            ) from exc
        return self.url

    def get_url(self) -> str:
        if self.url is None:
            self.url = _build_public_url(self.port)
        return self.url

    def get_pid(self) -> Optional[int]:
        return None if self.proc is None else self.proc.pid

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        finally:
            self.proc = None

