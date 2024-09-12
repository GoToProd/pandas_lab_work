import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

#Шаг 0: объединить все данные в 1 файл.
# Загрузка ваших CSV файлов
df_abgroup = pd.read_csv('ABgroup.csv', delimiter=',')
df_cash = pd.read_csv('Cash.csv', delimiter=',')
df_cheaters = pd.read_csv('Cheaters.csv', delimiter=',')
df_money = pd.read_csv('Money.csv', delimiter=',')
df_platforms = pd.read_csv('Platforms.csv', delimiter=',')

if 'user_id,group' in df_abgroup.columns:
    df_abgroup[['user_id', 'group']] = df_abgroup['user_id,group'].str.split(',', expand=True)
    df_abgroup = df_abgroup.drop(columns=['user_id,group'])

# Объединение данных по user_id
merged_data = df_abgroup \
    .merge(df_cash, on='user_id', how='left') \
    .merge(df_cheaters, on='user_id', how='left') \
    .merge(df_money, on='user_id', how='left') \
    .merge(df_platforms, on='user_id', how='left')

# Переименование колонок, чтобы избежать путаницы
merged_data = merged_data.rename(columns={
    'date_x': 'date_cash',  # Переименование колонки 'date' из Cash в 'date_cash'
    'date_y': 'date_money'  # Переименование колонки 'date' из Money в 'date_money'
})

# Переупорядочивание и выбор нужных колонок
merged_data = merged_data[['user_id', 'group', 'date_cash', 'cash', 'date_money', 'money', 'platform']]

# Сохранение объединённых данных в новый CSV файл с разделителем ;
merged_data.to_csv('MergedData.csv', sep=';', index=False)


# Шаг 1: Загрузка объединенного файла с указанием разделителя
merged_data = pd.read_csv('MergedData.csv', delimiter=';')

# Шаг 2: Проверка данных
# Удаление дублирующих записей (если есть)
merged_data = merged_data.drop_duplicates()

# Шаг 3: Фильтрация данных (например, удаление строк с отсутствующими значениями в ключевых столбцах)
merged_data_clean = merged_data.dropna(subset=['user_id', 'group', 'money'])

# Фильтрация по читерам (если данные о читерах включены)
# Например, если у вас есть поле 'cheaters' с значениями 1 и 0
if 'cheaters' in merged_data_clean.columns:
    cheaters_list = merged_data_clean[merged_data_clean['cheaters'] == 1]['user_id'].tolist()
    merged_data_clean = merged_data_clean[~merged_data_clean['user_id'].isin(cheaters_list)]

# Шаг 4: Анализ данных
# Расчет ARPU (Average Revenue Per User)
arpu = merged_data_clean.groupby('group')['money'].sum() / merged_data_clean.groupby('group')['user_id'].count()
print("ARPU:")
print(arpu)

# Расчет ARPPU (Average Revenue Per Paying User)
paying_users = merged_data_clean[merged_data_clean['money'] > 0]
arppu = paying_users.groupby('group')['money'].sum() / paying_users.groupby('group')['user_id'].count()
print("ARPPU:")
print(arppu)

# Построение графика суммарного дохода по группам и платформам
grouped = merged_data_clean.groupby(['group', 'platform'])['money'].sum().unstack()
print("Grouped Data for Plotting:")
print(grouped)

if not grouped.empty and grouped.select_dtypes(include='number').notnull().all().all():
    grouped.plot(kind='bar', figsize=(12, 6))
    plt.title('Total Revenue by Group and Platform')
    plt.xlabel('Group')
    plt.ylabel('Total Revenue')
    plt.legend(title='Platform')
    plt.show()
else:
    print("No numeric data to plot or data is empty.")

# Функция для расчета доверительных интервалов
def calculate_confidence_interval(data, confidence=0.95):
    if len(data) > 1:
        mean = data.mean()
        std_err = stats.sem(data)
        margin_of_error = std_err * stats.t.ppf((1 + confidence) / 2., len(data)-1)
        return mean - margin_of_error, mean + margin_of_error
    else:
        return float('nan'), float('nan')

# Расчет доверительных интервалов для ARPU
arpu_control = merged_data_clean[merged_data_clean['group'] == 'control']['money']
arpu_test = merged_data_clean[merged_data_clean['group'] == 'test']['money']

ci_control = calculate_confidence_interval(arpu_control)
ci_test = calculate_confidence_interval(arpu_test)

print("ARPU Control Group Confidence Interval:", ci_control)
print("ARPU Test Group Confidence Interval:", ci_test)

# Расчет доверительных интервалов для ARPPU
arppu_control = paying_users[paying_users['group'] == 'control']['money']
arppu_test = paying_users[paying_users['group'] == 'test']['money']

ci_arppu_control = calculate_confidence_interval(arppu_control)
ci_arppu_test = calculate_confidence_interval(arppu_test)

print("ARPPU Control Group Confidence Interval:", ci_arppu_control)
print("ARPPU Test Group Confidence Interval:", ci_arppu_test)
