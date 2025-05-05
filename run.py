import importlib

app_module = importlib.import_module("app.main.main")
app = getattr(app_module, "app")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
