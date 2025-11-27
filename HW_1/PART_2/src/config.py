from dataclasses import dataclass

ORIGINAL_FEATURES = [
    'name',
    'year',
    'km_driven',
    'fuel',
    'seller_type',
    'transmission',
    'owner',
    'mileage',
    'engine',
    'max_power',
    'torque',
    'seats'
]

@dataclass(frozen=True)
class AutoOptions:
    BRANDS = ['Maruti', 'Skoda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
        'Mahindra', 'Honda', 'Chevrolet', 'Fiat', 'Datsun', 'Tata', 'Jeep',
        'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW',
        'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
        'Kia', 'Force', 'Ambassador', 'Isuzu', 'Peugeot']

    FUELS = ['Diesel', 'Petrol', 'LPG', 'CNG']
    TRANSMISSIONS = ['Manual', 'Automatic']
    OWNER = ['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car']
    SELLER_TYPE = ['Individual', 'Dealer', 'Trustmark Dealer']

