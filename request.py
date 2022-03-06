import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'MonthlyIncome':20000, 'TypeOfContact':2, 'Occupation':3, 'CityTier':1, 'PreferredPropertyStar:4'})

print(r.json())