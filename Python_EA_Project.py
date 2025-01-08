import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

FCA = "AlxUp1O2bAhFZvGwyliAr3HKjQAgCj5gZ3ssCLXU"
FIXER = "KK34viuXxt0BVkCawOehcSgA9oZNarpf"
OPEN = "69a431341ce84f02a4a0a30e9b69702a"


class MarketData:
    def __init__(self, currency_pair, interval=1, duration=5):
        assert isinstance(interval, int) & interval > 0, 'Interval must be a non negative integer'
        assert isinstance(duration, int) & duration > 0, 'Duration must be a non negative integer less than 5 minutes'
        assert isinstance(currency_pair, str) and len(currency_pair) == 6, "Currency pair must be a string of length 6"

        self.currency_pair = currency_pair
        self.period = period
        self.interval = interval
        self.duration = duration

    def FCA_Live(self):
        base_url = f"https://api.freecurrencyapi.com/v1/latest?apikey={FCA}"
        params = {"base_currency": self.currency_pair[:3], "quote_currencies": self.currency_pair[3:]}
        end_time = datetime.now() + timedelta(seconds=self.duration)
        while datetime.now() < end_time:
            response = requests.get(base_url, params=params)
            data = response.json()
            return data['data'][self.currency_pair[3:]]
            time.sleep(interval)

    def FIXER_Live(self):
        base_url = "https://api.apilayer.com/fixer/latest"
        headers = {"apikey": f"{FIXER}"}
        params = {"symbols": self.currency_pair[3:], "base": self.currency_pair[:3]}
        end_time = datetime.now() + timedelta(seconds=self.duration)
        while datetime.now() < end_time:
            response = requests.get(base_url, params=params, headers=headers)
            data = response.json()
            return data['rates'][self.currency_pair[3:]]
            time.sleep(interval)

    def OPEN_Live(self):
        base_url = "https://openexchangerates.org/api/latest.json"
        params = {"app_id": f"{OPEN}", "symbols": self.currency_pair[:3]}
        end_time = datetime.now() + timedelta(seconds=self.duration)
        while datetime.now() < end_time:
            response = requests.get(base_url, params=params)
            data = response.json()
            return data['rates'][self.currency_pair[:3]]
            time.sleep(interval)


class RateInterpolate:
    def __init__(self, before, after):
        self.Before = before
        self.After = after
        self.growth_rates = (self.After[1:6] / self.Before[1:6]) ** (1 / 3) - 1
        self.Saturday = [(self.Before[0] + timedelta(days=1))]
        self.Sunday = [(self.Before[0] + timedelta(days=2))]

    def saturday(self):
        for n in range(len(self.growth_rates)):
            self.Saturday.append(self.Before[n + 1] * (1 + self.growth_rates[n]))
        return self.Saturday

    def sunday(self):
        for n in range(len(self.growth_rates)):
            self.Sunday.append(self.Saturday[n + 1] * (1 + self.growth_rates[n]))
        return self.Sunday

    def create_df(self):
        self.saturday()
        self.sunday()
        new_rows = {'Date': [self.Saturday[0], self.Sunday[0]],
                    'Close': [self.Saturday[1], self.Sunday[1]],
                    'Open': [self.Saturday[2], self.Sunday[2]],
                    'High': [self.Saturday[3], self.Sunday[3]],
                    'Low': [self.Saturday[4], self.Sunday[4]],
                    'Volume': [self.Saturday[5], self.Sunday[5]]}

        new_frame = pd.DataFrame(new_rows, columns=list(new_rows.keys()))
        return new_frame


class PriceAlgorithms:
    def __init__(self, market_data):
        self.data = market_data

    def twap(self, start, end):
        data = self.data[self.data['Date'].isin(pd.date_range(start, end))]
        twap = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        return twap.mean()

    def vwap(self, start, end):
        data = (self.data[self.data['Date'].isin(pd.date_range(start, end))]).copy()
        data['twap'] = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4
        vwap = (data['twap'] * data['Volume']).sum() / data['Volume'].sum()
        return vwap


class VolumeProfiler:
    def __init__(self, vol_data):
        self.data = vol_data
        self.day_pairs = {}

    def get_times(self):
        data = self.data
        data.set_index('timestamp', inplace=True)

        # Slice the April Intraday Data by day and store each day as a new DataFrame and remove missing weekends
        days = [group for _, group in data.groupby(pd.Grouper(freq='D'))]
        days = [vals for vals in days if not vals.empty]

        # For each day in April, store each hour that has intraday data as a new DataFrame
        for day in days:
            hours = ([group for _, group in day.groupby(pd.Grouper(freq='H'))])
            hours = [vals for vals in hours if not vals.empty]
            hour_pairs = {}

            # For each hour in the day, calculate the rolling average over 5 minutes
            for hour in hours:
                profile = hour["volume"].rolling(5).sum()

                # Save the highest and lowest rolling averages
                interval_max = np.argmax(profile)
                interval_min = np.argmin(profile)

                # Store the 5 minute interval with the highest and lowest sums as most liquid & most stale respectively
                liquid_interval = hour.iloc[interval_max - 4:interval_max + 1]
                stale_interval = hour.iloc[interval_min - 4:interval_min + 1]

                # Add the start time of the most liquid and most stale periods as values to the hour_pairs dictionary
                hour_pairs[f"hour {hour.index.hour[0]} liquid"] = liquid_interval.index[0]
                hour_pairs[f"hour {hour.index.hour[0]} stale"] = stale_interval.index[0]

            # Store all the key:value pairs for the days most and least liquid as values of the day_pairs dictionary
            self.day_pairs[f"April {day.index.day[0]}"] = hour_pairs


    def get_profile(self):
        self.get_times()
        high_times = {}
        low_times = {}

        time_of_day = 0
        while time_of_day < 24:
            times_high = []
            times_low = []
            for day_of_month in range(1, 23):
                if f'April {day_of_month}' not in self.day_pairs:
                    continue

                elif f'hour {time_of_day} liquid' not in self.day_pairs[
                    f'April {day_of_month}'] or f'hour {time_of_day} stale' not in self.day_pairs[f'April {day_of_month}']:
                    continue

                else:
                    # Create a list of the start times for peak liquidity
                    times_high.append(self.day_pairs[f'April {day_of_month}'][f'hour {time_of_day} liquid'].time())
                    times_low.append(self.day_pairs[f'April {day_of_month}'][f'hour {time_of_day} stale'].time())

                # Convert the times to seconds since midnight
            seconds_h = [datetime.combine(datetime.min, high) - datetime.min for high in times_high]
            seconds_h = [int(sec.total_seconds()) for sec in seconds_h]

            # Calculate the average number of seconds
            avg_seconds_h = sum(seconds_h) // len(seconds_h)

            # Convert the result back to a time object
            avg_time_h = (datetime.min + timedelta(seconds=avg_seconds_h)).time()
            high_times[f"hour {time_of_day}"] = avg_time_h

            seconds_l = [datetime.combine(datetime.min, low) - datetime.min for low in times_low]
            seconds_l = [int(sec.total_seconds()) for sec in seconds_l]

            # Calculate the average number of seconds
            avg_seconds_l = sum(seconds_l) // len(seconds_l)

            # Convert the result back to a time object
            avg_time_l = (datetime.min + timedelta(seconds=avg_seconds_l)).time()
            low_times[f"hour {time_of_day}"] = avg_time_l
            time_of_day += 1

        return high_times, low_times


class OrderExecution:
    def __init__(self, market):
        self.live = run_price_comparison()[0][market]

    def demo_execution(self, currency_pair, direction, size, demo=True):
        if not demo:
            api_key = "NA"
            api_secret = "NA_NA"
            api_url = "https://api-demo.forex.com/v3/accounts"

            auth_response = requests.post(f"{api_url}/auth", json={"apiKey": api_key, "apiSecret": api_secret})
            auth_response.raise_for_status()
            access_token = auth_response.json()["accessToken"]

            headers = {"Authorization": f"Bearer {access_token}"}
            order_response = requests.post(f"{api_url}/orders",
                                           json={
                                               "orderType": "market",
                                               "currencyPair": currency_pair,
                                               "direction": direction,
                                               "size": size
                                           },
                                           headers=headers
                                           )
            order_response.raise_for_status()

        else:
            return

    def twap_execution(self, ref, direction, size, duration=3600):
        """TWAP (Time Weighted Average Price) follows a linear schedule to execute an order evenly over a specified timeperiod.
        It aims to minimise slippage / variance against a TWAP reference."""

        twap_reference = ref
        duration = timedelta(seconds=duration)
        end_time = datetime.now() + duration
        total_quantity = size
        currency = currency_pair
        quantity_remaining = total_quantity
        while datetime.now() < end_time and quantity_remaining > 0:
            time_remaining = end_time - datetime.now()
            target_quantity = total_quantity * (1 - time_remaining / duration)
            quantity_to_execute = target_quantity - (total_quantity - quantity_remaining)
            for i in range(len(self.live)):
                if abs(twap_reference - self.live[i]) > (self.live[i] * 0.05):
                    time.sleep(time_remaining.seconds / 100)
                    continue
                else:
                    self.demo_execution(currency, direction, quantity_to_execute)
                    quantity_remaining -= quantity_to_execute
                    time.sleep(duration.seconds / total_quantity)

    def vwap_execution(self, ref, profile, direction, size, duration=3600):
        """VWAP (Volume Weighted Average Price) execute an order over a specified time period, following the historical volume
        reference price to execute a larger portion of the order when the market is more liquid. """

        vwap_referance = ref
        vwap_profile = profile
        duration = timedelta(seconds=duration)
        end_time = datetime.now() + duration
        total_quantity = size
        currency = currency_pair
        quantity_remaining = total_quantity
        while datetime.now() < end_time and quantity_remaining > 0:
            time_remaining = end_time - datetime.now()
            target_quantity = total_quantity * (1 - time_remaining / duration)
            quantity_to_execute = target_quantity - (total_quantity - quantity_remaining)
            for i in range(len(self.live)):
                if abs(vwap_referance - self.live[i]) > (self.live[i] * 0.05):
                    time.sleep(time_remaining.seconds / 100)
                    continue
                else:
                    target_time = vwap_profile[f'hour {datetime.now().hour}']
                    t_delta = datetime.combine(datetime.min, target_time) - datetime.combine(datetime.min,
                                                                                         datetime.now().time())
                    min_to_target = t_delta.total_seconds() / 60
                    if abs(min_to_target) < 5:
                        multiplier = 1.3
                    elif 5 < abs(min_to_target) < 15:
                        multiplier = 1.1
                    else:
                        multiplier = 0.8
                    quantity_to_execute *= multiplier
                    self.demo_execution(currency, direction, quantity_to_execute)
                    quantity_remaining -= quantity_to_execute
                    time.sleep(duration.seconds / total_quantity)


def clean_data(data):
    # Rename and drop any columns
    data = data.rename(columns={"Vol.": "Volume", "Price": "Close"})
    data = data.drop(columns="Change %")

    # Convert 'k' string to 1*e3, using regex to evaluate this as * 1000 to convert 100k to 100000
    data['Volume'] = data['Volume'].replace({'K': '*1e3'}, regex=True).map(pd.eval).astype(int)

    # Organize the data as oldest first
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.sort_values(by='Date', inplace=True)

    # Use date shift to determine where the gap between days is more than 1, signifying a weekend data gap
    # Make a copy for comparing raw vs interpolated data later on
    data['Date_Diff'] = (data['Date'] - data['Date'].shift()).dt.days
    interpolated_data = data.copy(deep=True)
    for i in range(len(data)):
        if data.iloc[i]['Date_Diff'] > 1.0:
            Before = data.iloc[i - 1]
            After = data.iloc[i]
            interpolated_data = pd.concat([interpolated_data, (RateInterpolate(Before, After).create_df())],
                                          ignore_index=True)
        else:
            pass

    # Re-organize by date now that new values have been added, and round off the accuracy to 4dp for rates, and 0dp for volume
    interpolated_data.sort_values(by='Date', inplace=True)
    data = data.round(4)
    interpolated_data = interpolated_data.round(4)
    data = data.round({"Volume": 0})
    interpolated_data = interpolated_data.round({"Volume": 0})

    # Drop the now redundant date_diff column and return the cleaned raw and cleaned interpolated data
    data = data.drop(columns="Date_Diff")
    interpolated_data = interpolated_data.drop(columns="Date_Diff")
    interpolated_data = interpolated_data.drop(labels=729, axis=0)
    return data, interpolated_data


EUR_data = pd.read_parquet("EUR_USD 2Yr Historic", engine="fastparquet")
VolProfile = pd.read_parquet("AprilData.parquet", engine="fastparquet")
profile = VolumeProfiler(VolProfile).get_profile()
cleaned_EUR_data_int = clean_data(EUR_data)[1]
cleaned_EUR_data = clean_data(EUR_data)[0]
currency_pair = 'EURUSD'
period = 30  # in days
start = datetime.now() - timedelta(days=period)
end = datetime.now()
start = start.strftime("%Y-%m-%d")
end = end.strftime("%Y-%m-%d")


def test_price_algos_eur():
    market_data = cleaned_EUR_data
    TWAP = round(PriceAlgorithms(market_data).twap(start, end), 4)
    VWAP = round(PriceAlgorithms(market_data).vwap(start, end), 4)
    return TWAP, VWAP


def test_price_algos_eur_int():
    market_data = cleaned_EUR_data_int
    TWAP = round(PriceAlgorithms(market_data).twap(start, end), 4)
    VWAP = round(PriceAlgorithms(market_data).vwap(start, end), 4)
    return TWAP, VWAP


def run_price_comparison():
    live_rates = {
        'FCA': [round(MarketData(currency_pair).FCA_Live(), 5)],
        'FIXER': [round(MarketData(currency_pair).FIXER_Live(), 5)],
        'OPEN': [round(1 / (MarketData(currency_pair).OPEN_Live()), 5)]
    }
    twap_delta = {f'{key}_delta_to_twap': live - test_price_algos_eur()[0] for (key, live) in live_rates.items()}
    vwap_delta = {f'{key}_delta_to_vwap': live - test_price_algos_eur()[1] for (key, live) in live_rates.items()}
    twap_delta_int = {f'{key}_delta_to_int_twap': live - test_price_algos_eur_int()[0] for (key, live) in
                      live_rates.items()}
    vwap_delta_int = {f'{key}_delta_to_int_vwap': live - test_price_algos_eur_int()[1] for (key, live) in
                      live_rates.items()}
    return live_rates, twap_delta, twap_delta_int, vwap_delta, vwap_delta_int


def get_volume_profile():
    vwap_best, vwap_worst = profile[0], profile[1]
    return vwap_best, vwap_worst


def run_mock_execution():
    order = OrderExecution('FCA')
    # test twap w/ twap raw pricing
    order.twap_execution(ref=test_price_algos_eur()[0], direction=1, size=100000, duration=10)

    # test twap w/ interpolated pricing
    order.twap_execution(ref=test_price_algos_eur_int()[0], direction=1, size=100000, duration=10)

    # test vwap w/ vwap raw pricing
    order.vwap_execution(ref=test_price_algos_eur()[1], direction=1, size=100000, profile=get_volume_profile()[0], duration=10)

    # test vwap w/ interpolated pricing
    order.vwap_execution(ref=test_price_algos_eur_int()[1], direction=1, size=100000, profile=get_volume_profile()[0], duration=10)
    return 'pass'


print(f'30day TWAP = {test_price_algos_eur()[0]}, 30day VWAP = {test_price_algos_eur()[1]} \n')
print(
    f'30day TWAP with interpolation = {test_price_algos_eur_int()[0]}, 30day VWAP with interpolation = {test_price_algos_eur_int()[1]} \n')
comp = [item for item in run_price_comparison()]
for i in range(len(comp[0])):
    for j in range(len(comp)):
        print(list(comp[j].keys())[i], list(comp[j].values())[i])
    print("\n")
print(run_mock_execution())
