from urllib.request import urlopen
from urllib.parse import urljoin
import json
from io import BytesIO
from zipfile import ZipFile as zip
from tqdm import tqdm

url = "https://data.votchallenge.net/vot2018/main/description.json"
out = "vot2018.zip"

with zip(out, "w") as outzip:
    for seq in tqdm(json.load(urlopen(url))["sequences"]):
        surl = seq["channels"].popitem()[1]["url"]
        with urlopen(urljoin(url, surl)) as f, BytesIO(f.read()) as b, zip(b) as z:
            for name in z.namelist():
                outzip.writestr(seq["name"] + "/" + name, z.read(name))
