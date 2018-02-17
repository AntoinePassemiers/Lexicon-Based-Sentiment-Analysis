import requests, zipfile, io
r = requests.post("http://mpqa.cs.pitt.edu/request_resource.php", data={"name":"bla", "organization":"yoyo", "email":"bla@yoyo.com", "dataset":"subj_lexicon"})
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall()

#props to https://stackoverflow.com/a/14260592
