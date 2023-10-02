import pandas as pd
import numpy as np

# create a class called SolarData -- when it is initialized with certain params it will generate the data for us

class SolarData:
    def __init__(self, tilt, dc_rating, system_loss=0.14, inverter_efficiency=0.95):

        self.PANEL_TILT = tilt
        self.INVERTER_EFFICIENCY = inverter_efficiency
        self.SYSTEM_LOSS = system_loss
        self.DC_SYSTEM_SIZE = dc_rating

        # elevation angle is 90 - (solar zenith angle)
        # estimate global tilted irradiance by:
        # GTI = direct normal irradiance (DNI) * sin(elevation + panel tilt) + diffuse horizontal irradiance (DHI)

        # panels are rated according to peak solar hours (PSH), which corresponds to a solar irradiance of 1000 W/m^2
        # if we compute GTI, we can estimate the percentage of this peak output that a panel will produce (minus 14% system losses)

        df = pd.read_csv('caltech_solar_radiation.csv', parse_dates=True)

        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        df["datetime"] = df["datetime"].dt.floor("H")
        df.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1, inplace=True)
        df.set_index('datetime', inplace=True)

        df['Solar Elevation Angle'] = 90 - df['Solar Zenith Angle']

        df['GTI'] = df['DNI'] * np.sin(np.radians(df['Solar Elevation Angle'])) + df['DHI']
        
        df['Solar Generation (kWh)'] = dc_rating * (df['GTI'] / 1000) * (1-system_loss) * inverter_efficiency

        self.df = df

    def get_generation_data(self):
        return self.df['Solar Generation (kWh)']
    
    def get_full_data(self):
        return self.df