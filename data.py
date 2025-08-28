import requests, zipfile, io, os

def download_data(url: str, name: str):
    api_url = (
        "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        f"?public_key={url}"
    )
    download_url = requests.get(api_url).json()["href"]
    
    with requests.get(download_url, stream=True) as r, open(f"{name}.zip", "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    os.makedirs(name, exist_ok=True)
    with zipfile.ZipFile(f"{name}.zip", "r") as z:
        z.extractall(name)