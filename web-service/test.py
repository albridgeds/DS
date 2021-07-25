import requests

url = 'http://localhost:5000/results'

test_data_0 = {'calls': 40.0, 'minutes': 311.90, 'messages': 83.0, 'mb_used': 19915.42}
r = requests.post(url,json=test_data_0)
print(r.json())

test_data_1 = {'calls': 19.0, 'minutes': 113.53, 'messages': 158.0, 'mb_used': 15616.05}
r = requests.post(url,json=test_data_1)
print(r.json())