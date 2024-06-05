import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import numpy as np





# Laden der Daten
file_path = 'lstm_train/raw_data_last_year.csv'
data = pd.read_csv(file_path)
data.replace('undefined', np.nan, inplace=True)
data["battery_current"] = pd.to_numeric(data['battery_current'], errors='coerce')
data["battery_voltage"] = pd.to_numeric(data['battery_voltage'], errors='coerce')
data.dropna(inplace=True)


# Bestimmung aller Maxima und Minima basierend auf den festgelegten Bedingungen
voltage_high_threshold = 55  # 100% SoC
voltage_low_threshold = 46.5  # 0% SoC
current_low_threshold = 2  # Niedriger Strom für beide Zustände

# Bedingungen für 100% SoC und 0% SoC
condition_soc_100 = (data['battery_voltage'] >= voltage_high_threshold) & (data['battery_current'].abs() <= current_low_threshold)
condition_soc_0 = (data['battery_voltage'] <= voltage_low_threshold) & (data['battery_current'].abs() <= current_low_threshold)

# Alle relevanten Zeitpunkte für 100% und 0% SoC
times_soc_100_all = data[condition_soc_100][['Time', 'battery_voltage', 'battery_current']]
times_soc_0_all = data[condition_soc_0][['Time', 'battery_voltage', 'battery_current']]

# Gruppieren von benachbarten Maxima und Minima
def group_points(df, gap):
    groups = []
    current_group = []
    last_time = None

    for index, row in df.iterrows():
        if last_time is None or (row['Time'] - last_time) <= gap:
            current_group.append(row)
        else:
            groups.append(current_group)
            current_group = [row]
        last_time = row['Time']

    if current_group:
        groups.append(current_group)
    
    return groups

def select_extreme_points(groups, select_max=True):
    selected_points = []
    for group in groups:
        if select_max:
            selected_points.append(max(group, key=lambda x: x['battery_voltage']))
        else:
            selected_points.append(min(group, key=lambda x: x['battery_voltage']))
    return selected_points

# Definieren des Abstands (in ms) zur Gruppierung benachbarter Punkte
gap = 60 * 60 * 1000  # 1 Stunde

# Gruppieren und Auswählen von Extrempunkten
groups_100 = group_points(times_soc_100_all, gap)
selected_100 = select_extreme_points(groups_100, select_max=True)

groups_0 = group_points(times_soc_0_all, gap)
selected_0 = select_extreme_points(groups_0, select_max=False)

# Erstellen von DataFrames aus den ausgewählten Punkten
selected_100_df = pd.DataFrame(selected_100)
selected_0_df = pd.DataFrame(selected_0)

# Funktion zur Berechnung der Restkapazität zwischen zwei Zeitpunkten
def calculate_capacity_between_points(start_time, end_time):
    data_segment = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]
    time_hours = (data_segment['Time'] - data_segment['Time'].iloc[0]) / (1000 * 3600)  # Zeit in Stunden
    capacity = simpson(y=data_segment['battery_current'], x=time_hours)  # Integration des Stroms über die Zeit
    print("-------------")
    print(pd.to_datetime(start_time, unit='ms'))
    print(pd.to_datetime(end_time, unit='ms'))
    print(data_segment)
    print(time_hours)
    print(capacity)
    print()
    return capacity

# Liste zur Speicherung der Kapazitäten
filtered_capacities = []

# Überprüfung der Reihenfolge der Punkte und Berechnung der Restkapazität zwischen aufeinanderfolgenden 100%- und 0%-Zeitpunkten
for i in range(len(selected_100_df)):
    start_time = selected_100_df.iloc[i]['Time']
    end_time_candidates = selected_0_df[selected_0_df['Time'] > start_time]
    if not end_time_candidates.empty:
        end_time = end_time_candidates['Time'].min()
        next_max = selected_100_df[selected_100_df['Time'] > start_time]
        if next_max.empty or end_time < next_max['Time'].min():
            capacity = calculate_capacity_between_points(start_time, end_time)
            filtered_capacities.append({'start_time': start_time, 'end_time': end_time, 'capacity': capacity})

# Erstellen eines DataFrames aus den gefilterten Kapazitätsberechnungen
filtered_capacities_df = pd.DataFrame(filtered_capacities)
filtered_capacities_df['abs_capacity'] = filtered_capacities_df['capacity'].abs()

# Konvertieren der Zeitstempel in ein lesbares Datumsformat
filtered_capacities_df['start_time'] = pd.to_datetime(filtered_capacities_df['start_time'], unit='ms')
filtered_capacities_df['end_time'] = pd.to_datetime(filtered_capacities_df['end_time'], unit='ms')

# Visualisierung der relevanten Zeitpunkte für 100% und 0% SoC auf der Grafik mit größeren Markierungen
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(data['Time'], data['battery_voltage'], label='Battery Voltage')
plt.scatter(selected_100_df['Time'], selected_100_df['battery_voltage'], color='red', s=10, label='100% SoC (All)')
plt.scatter(selected_0_df['Time'], selected_0_df['battery_voltage'], color='blue', s=10, label='0% SoC (All)')
plt.scatter(filtered_capacities_df['start_time'].astype(np.int64) // 10**6, [voltage_high_threshold] * len(filtered_capacities_df), color='darkred', s=100, label='Selected 100% SoC')
plt.scatter(filtered_capacities_df['end_time'].astype(np.int64) // 10**6, [voltage_low_threshold] * len(filtered_capacities_df), color='darkblue', s=100, label='Selected 0% SoC')
plt.ylabel('Voltage (V)')
plt.title('Battery Voltage over Time')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data['Time'], data['battery_current'], label='Battery Current', color='orange')
plt.scatter(selected_100_df['Time'], selected_100_df['battery_current'], color='red', s=10, label='100% SoC (All)')
plt.scatter(selected_0_df['Time'], selected_0_df['battery_current'], color='blue', s=10, label='0% SoC (All)')
plt.scatter(filtered_capacities_df['start_time'].astype(np.int64) // 10**6, [current_low_threshold] * len(filtered_capacities_df), color='darkred', s=100, label='Selected 100% SoC')
plt.scatter(filtered_capacities_df['end_time'].astype(np.int64) // 10**6, [current_low_threshold] * len(filtered_capacities_df), color='darkblue', s=100, label='Selected 0% SoC')
plt.ylabel('Current (A)')
plt.xlabel('Time')
plt.title('Battery Current over Time')
plt.legend()

plt.tight_layout()
plt.show()

print(filtered_capacities_df)