from fastapi import FastAPI

app = FastAPI(title="AI Server")


@app.get("/health")
def health():
    return {"status": "ok", "message": "CI/CD test success!"}


@app.get("/gpu")
def gpu_status():
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
        return {"cuda_available": False}
    except ImportError:
        return {"cuda_available": False, "error": "torch not installed"}


@app.get("/chromadb")
def chromadb_status():
    try:
        import chromadb

        client = chromadb.Client()
        return {"chromadb": "ok", "heartbeat": client.heartbeat()}
    except ImportError:
        return {"chromadb": "not installed"}
