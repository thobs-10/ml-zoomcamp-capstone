import requests
					
input_data = {
    # 'id': 1223,
    'Gender': 'Male',
    # 'Customer Type': 'Loyal Customer',
    'Age': 36,
    # 'Type of Travel':'Business Travel',
    # 'Class': 'Business',
    'Flight Distance': 357,
    'Inflight wifi service': 3,
    'Departure/Arrival time convenient': 3,
    'Ease of Online booking': 2,
}

response = requests.post(
    url='http://127.0.0.1:8000/predict',
    json=input_data,
    headers={'Content-Type': 'application/json'}
)
print(response.status_code)
print(response.json())
if response.json()['data'] == 0:
    print('Customer satisfaction: dissatisfied or neutral')
else:
    print('Customer satisfaction: satisfied')