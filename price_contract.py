import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

np.seed = 12121  # reproducibility


class LinearModel:
    def __init__(self, slope, intercept):
        self.intercept = intercept
        self.slope = slope

    @staticmethod
    def gen_noise(max_noise, min_noise):
        return np.random.uniform(low=min_noise, high=max_noise)

    @staticmethod
    def extrapolate_seasonality(data, seasonal_cycle,
                                start_date,
                                seasonality=12):
        # starting date index
        start_idx = pd.Index(data['Dates']).get_loc(start_date)

        # corresponding seasonal indices
        season_idx = start_idx % seasonality  # assuming a seasonality of 12 months

        # corresponding seasonal values
        seasonal_value = seasonal_cycle.iloc[season_idx]

        return seasonal_value

    # predicting the base value from the trend component
    def predict_base(self, ordinal_date):
        return self.slope * ordinal_date + self.intercept

    # predict the price at a given date
    def predict(self, date):
        base_value = self.predict_base(date_to_ordinal(date))
        noise_value = self.gen_noise(max_noise, min_noise)
        seasonal_value = self.extrapolate_seasonality(ng_data, s, date)

        return base_value + noise_value + seasonal_value


# convert a string date or datetime object to ordinal date
def date_to_ordinal(date=None, date_format='%Y-%m-%d'):
    # Convert string date to datetime object
    if type(date) is str:
        date_obj = datetime.strptime(date, date_format)
        # Convert datetime object to ordinal date
        ordinal_date = date_obj.toordinal()
        return ordinal_date

    if type(date) is not str:
        return date.toordinal()


def get_trend_line(data, period=12):
    # Converting the dates into ordinal format
    o_dates = pd.DataFrame({'Dates': [date_to_ordinal(date) for date in data['Dates']]})

    # Getting the trend component
    t, s, r = seasonal_decomposition(data['Prices'], period=period)

    # Trend data
    X = o_dates.iloc[period // 2: -1 - (period // 2 - 1)]
    y = t.dropna()
    X = X.to_numpy().reshape(-1, 1)
    y = y.to_numpy().reshape(-1, 1)

    # Linear regression model for the trend line
    lm = LinearRegression().fit(X, y)
    slope = lm.coef_[0][0]  # trend line gradient
    intercept = lm.intercept_  # trend line y-intercept

    return slope, intercept


def get_data():
    root = os.getcwd()
    price_data = pd.read_csv(os.path.join(root, 'Nat_Gas.csv'))
    price_data['Dates'] = pd.to_datetime(price_data['Dates'])
    return price_data


def seasonal_decomposition(data, model='additive', period=12):
    result = seasonal_decompose(data, model=model, period=period)
    return result.trend, result.seasonal, result.resid


def contract_price(injection_dates=None,
                   withdrawal_dates=None,
                   amount=None,
                   storage_rate=None,
                   maximum_storage=None,
                   storage_cost=None,
                   transport_cost=None,
                   pricing_model=None):
    """
    -----------------------------------------
    CONSTRAINTS:
    1. ALL GAS PURCHASED CAN BE STORED INSTANTLY AND GAS WITHDRAWN CAN BE SOLD INSTANTLY.
    2. INJECTION AND WITHDRAWAL DATE CANNOT BE ON THE SAME DAY
    3. STORAGE RATE IS CONSTANT FOR THE ENTIRE PERIOD
    4. STORAGE COST IS FIXED FOR THE ENTIRE PERIOD
    5. TRANSPORT COST IS FIXED FOR THE ENTIRE PERIOD
    6. THE BUYER CANNOT BUY MORE GAS THAN AVAILABLE STORAGE
    7. THE BUYER STARTS WITH 0 GAS IN STORAGE
    8. INJECTION AND WITHDRAWAL DATES MUST BE IN CHRONOLOGICAL ORDER
    9. THE FIRST INJECTION AND WITHDRAWAL DATE PAIR MUST BE VALID (i.e. INJECTION DATE < WITHDRAWAL DATE)
    10. THE BUYER WILL ALWAYS INJECT/WITHDRAW A FIXED AMOUNT OF GAS
    11. THE BUYER WILL ALWAYS WITHDRAW GAS AT SOME POINT AFTER AN INJECTION
    -----------------------------------------
    injection_dates:  a list of datetime objects at which gas is to be stored
    withdrawal_dates:  a list of datetime objects at which gas is to be withdrawn
    amount: amount of natural gas purchased in millions of MMBtu, and the amount injected/withdrawn. Stays constant.
    storage_rate: cost per million MMBtu in USD
    maximum_storage: maximum storage capacity in million MMBtu's
    storage_cost: fixed monthly storage fee in USD
    transport_cost: one-off fee per transportation journey in USD
    pricing_model: provide a gas pricing model to extrapolate OOD

    Note that the total contract value is given by:
    contract_val = sell_date_price - buy_date_price - total_costs
    total_costs = injection_date_cost + withdrawal_date_cost + storage_cost + transport_cost + storage_rate * amount
    """
    # Checking validity of parameters
    if amount > maximum_storage:
        raise ValueError("Cannot purchase more than storage capacity")
    if amount < 0 or maximum_storage < 0:
        raise ValueError("Cannot purchase negative amount of gas")
    if storage_rate < 0:
        raise ValueError("Storage rate cannot be negative")
    if transport_cost < 0:
        raise ValueError("Transport cost cannot be negative")
    if pricing_model is None:
        raise ValueError("No pricing model provided")
    if storage_cost < 0:
        raise ValueError("Storage cost cannot be negative")

    # Sorting the dates
    injection_dates.sort()
    withdrawal_dates.sort()

    # Check validity of injection_dates and withdrawal_dates
    if len(injection_dates) != len(withdrawal_dates):
        raise ValueError("Number of injection dates must be equal to number of withdrawal dates")
    if not injection_dates:
        raise ValueError("Injection dates cannot be empty")
    if withdrawal_dates[0] < injection_dates[0]:
        raise ValueError("The first withdrawal date cannot be before the first injection date")
    for i in range(1, len(injection_dates)):
        if injection_dates[i] < withdrawal_dates[i - 1]:
            raise ValueError("Cannot have two withdrawals between any two injections")

    buying_prices = []
    selling_prices = []

    total_transport_cost = transport_cost * (len(injection_dates) + len(withdrawal_dates))
    total_storage_rate_cost = (len(injection_dates) + len(withdrawal_dates)) * (storage_rate * amount)

    for id in injection_dates:
        # price is given per MMBtu
        buying_prices.append(pricing_model.predict(id)*amount*1e6 if not date_in_distribution(id) else ng_data['Prices'][ng_data['Dates'] == id].iloc[0]*amount*1e6)
    for wd in withdrawal_dates:
        selling_prices.append(pricing_model.predict(wd)*amount*1e6 if not date_in_distribution(wd) else ng_data['Prices'][ng_data['Dates'] == wd].iloc[0]*amount*1e6)

    # total cost of storage
    total_storage_cost = 0
    for inj, wd in zip(injection_dates, withdrawal_dates):
        inj_date = datetime.strptime(inj, "%Y-%m-%d")
        wd_date = datetime.strptime(wd, "%Y-%m-%d")
        delta = relativedelta(wd_date, inj_date)
        months_between = delta.years * 12 + delta.months
        total_storage_cost += months_between * storage_cost

    # Printing cash flows
    print(f"Buying prices: {buying_prices}")
    print(f"Selling prices: {selling_prices}")
    print(f"Total storage cost: {total_storage_cost}")
    print(f"Total transport cost: {total_transport_cost}")
    print(f"Total storage rate cost: {total_storage_rate_cost}")


    # contract price in USD
    return sum(selling_prices) - sum(buying_prices) - total_storage_cost - total_transport_cost - total_storage_rate_cost


def date_in_distribution(date):
    date = pd.Timestamp(date)
    if date in distribution_dates:
        return True
    else:
        return False


def visualise():
    # TO DO
    pass


if __name__ == '__main__':
    ng_data = get_data()  # natural gas data

    # time series decomposition
    decomposition_period = 12
    t, s, r = seasonal_decomposition(ng_data['Prices'], period=decomposition_period)
    m, c = get_trend_line(ng_data, period=decomposition_period)

    # decomposition parameters
    max_noise, min_noise = r.max(), r.min()

    # noise component
    noise = LinearModel.gen_noise(max_noise, min_noise)

    # Linear model from task 1 will no longer extrapolate 1 year into the future
    # It will simply predict the price at any given date (in ordinal format)
    model = LinearModel(m, c)

    # start date must be in the format 'YYYY-MM-DD'
    distribution_dates = list(ng_data['Dates'])

    # contract price inputs
    i_dates = ['2020-10-31', '2021-03-31']  # list of injection dates in the format 'YYYY-MM-DD' (string)
    w_dates = ['2021-01-31', '2022-01-31']  # list of withdrawal dates
    temp_test = pd.Timestamp(i_dates[0])

    # contract price parameters
    storage_rate = 15000  # storage rate in USD per million MMBtu
    amount = 200 # amount of gas purchased in millions of MMBtu
    transport_cost = 50000  # cost per transportation in USD
    storage_cost = 100000  # monthly rent in USD
    max_storage = 1000  # maximum storage capacity in millions of MMBtu
    temp = relativedelta(datetime.strptime(w_dates[0], '%Y-%m-%d'), datetime.strptime(i_dates[0], '%Y-%m-%d'))
    delta = temp.years * 12 + temp.months  # storage time in months between dates

    # get the contract price
    contract_price = contract_price(i_dates, w_dates, amount, storage_rate, max_storage, storage_cost, transport_cost, model)
    print("Contract price: ${:.2f}".format(contract_price))
