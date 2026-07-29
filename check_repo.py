import urllib.request, json

def list_folder(path):
    url = f"https://api.github.com/repos/klainfo/DefectData/contents/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req).read().decode())
    for item in data:
        print(item["type"], item["path"])
        if item["type"] == "dir":
            list_folder(item["path"])

list_folder("inst")
