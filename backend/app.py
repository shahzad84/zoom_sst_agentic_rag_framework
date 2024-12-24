from fastapi import FastAPI


app=FastAPI()


@app.get("/")
def main():
    print("hii")


# uvicorn app:app --reload