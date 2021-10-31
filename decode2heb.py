import zipfile
import json


archive = zipfile.ZipFile('data/ParaShoot.zip', 'r')
for f in ["train", "dev", "test"]:
    txtdata = archive.read(f'ParaShoot-main/{f}.json')
    json_out = json.loads(txtdata)
    with open(f'{f}.json', 'w') as fp:
        json.dump(json_out, fp,  ensure_ascii=False)


