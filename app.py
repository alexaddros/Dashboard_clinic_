import dash
from dash_table.Format import Format
import dash_table
import dash_html_components as html
import pandas as pd
import numpy as np


#Подгрузим данные и размножим их до 1000
data = pd.read_csv('./data/Statistika_Po_Laboratorii_Vrt_first_sheet.csv', encoding='windows-1251', decimal=",")
data.dropna(inplace=True, how='all',thresh=15)
data.dropna(inplace=True, how='all', axis=1) ##

duplicated_df = []
for i in range(102):
    duplicated_df.append(data)

full_df = pd.concat(duplicated_df)
full_df['год рождения'] -= np.random.randint(0, 19, full_df.shape[0])
full_df['Возраст'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Кол-во ооцитов на ЭКО 1 лунка'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Количество эмбрионов'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Количество бластоцист'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Количество Бц исп'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Целевое значение NTMSC на клетку'] = np.random.randint(1600, 2100, full_df.shape[0])
full_df['Вероятность_Беременности'] = 0
full_df['Прогноз_Беременности'] = np.random.choice([0, 1], full_df.shape[0], p=[1./4, 3./4])
full_df['Вероятность_Беременности'] = round(abs(full_df['Прогноз_Беременности'] - np.random.choice([0.1, 0.3, 0.26, 0.18, 0.15],
                                                full_df.shape[0], p=[1./10, 1./10, 1./5, 2./5, 1./5])),2)



full_df = full_df.sample(50).copy()  # 1000

num_cols = ['№ попытки', 'Количество фолликулов',
       'Получено всего ооцитов', 'Количество криоконсервированных ооцитов',
       #'Количество размороженных ооцитов',
        'Сперма донорская', 'Объем',
       'Концентрация', 'PROGR', 'NON PROGR', 'IMMOBIL', 'Норма',
       'Концентрация спермы (после обработки)',
       'Целевое значение NTMSC на клетку', 'Количество клеток для ИКСИ',
       'Кол-во ооцитов на ЭКО 1 лунка', 'Кол-во ооцитов в  инсемин. ЭКО',
       'Кол-во 2PN в ЭКО', 'Количество МII', 'Количество MI', 'Количество GV',
       'Количество DEG', 'Кол-во 2pn в ICSI', 'Неопл', 'Кол-во DEG',
       'Дисморфизм ооцитов', 'Количество эмбрионов', 'Количество бластоцист',
       'Количество Бц исп', 'ЕТ эмб-ов', 'Кол-во крио эмб-в',]


col_means_dict = {}
for col in num_cols:
    col_means_dict[col] = [full_df[col].mean(), full_df[col].std()]
df = full_df.copy()


external_stylesheets = [ {"href": "https://fonts.googleapis.com/css2?"
                        "family=Lato:wght@400;700&display=swap",
                        "rel": "stylesheet"},]


#Запуск самого Дашборда
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True  #Поработать с этим позже


#Элемент с таблицей. Вынесен в отдельную функцию, чтобы не дублировать код
def get_table(dff):
    return dash_table.DataTable(
            id='datatable-interactivity',
            columns=[ {'name': i, 'id': i, 'deletable': True, 'selectable': True, 'format': Format(nully='N/A')} for i in dff.columns],
            data=dff.to_dict('records'), style_cell={'whiteSpace': 'normal', 'height': 'auto'}, editable=True,
            filter_action='native', #sort_action='native',
            sort_mode='multi', #column_selectable='multi',
            #row_selectable=False, row_deletable=True,
            selected_columns=[], selected_rows=[], page_action='native',
            page_current=0, page_size=50,

            #Ошибка возникает тут. Если закоментить tooltip_conditional, то ошибки нет
            tooltip_conditional = [{'if': {'filter_query': '{{{}}} is nil'.format(col), 'column_id': col},
                         'type': 'markdown', 'value': 'Пустая ячейка'} for col in dff.columns]
            )

server = app.server
app.layout = html.Div([
        html.Div(
        children=[
        html.Div( children=[
        html.Img(src=app.get_asset_url('my_image_clinic.png')), # Подгружаем картинку
        html.H1(children="Pregnancy Analytics", className="header-title", style={"fontSize": "48px", 'color':'black'},),]),

        #Таблица
        html.Div(id='table-filtered',
                 children=[get_table(df)]), ]) ])



if __name__ == '__main__':
    app.run_server(debug=True, port=8050)