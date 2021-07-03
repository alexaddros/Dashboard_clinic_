import base64
import datetime
import io
import dash
from dash_core_components.RadioItems import RadioItems
from dash_html_components import Button
from dash_html_components.Div import Div
from dash_table.Format import Format
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np


class AllNums:
    def __init__(self):
        pass

    def __eq__(self, o):
        return True
        

# Подгрузим данные и размножим их до 1000
data = pd.read_csv('./data/Statistika_Po_Laboratorii_Vrt_first_sheet.csv', encoding='windows-1251', decimal=",")
# data = pd.read_csv('data/demo_first_sheet.csv', encoding='windows-1251')
# data = pd.read_excel('Тест_данных_по_клинике_для_Саши_май5.xlsx')
data.dropna(inplace=True, how='all',thresh=15)
data.dropna(inplace=True, how='all', axis=1) ##
duplicated_df = []
for i in range(102):
    duplicated_df.append(data)
full_df = pd.concat(duplicated_df)
# full_df.dropna(inplace=True, how='all', axis=0)
full_df['год рождения'] -= np.random.randint(0, 19, full_df.shape[0])
full_df['Возраст'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Кол-во ооцитов на ЭКО 1 лунка'] += np.random.randint(0, 19, full_df.shape[0])
#full_df['Количество криоконсервированных ооцитов'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Количество эмбрионов'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Количество бластоцист'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Количество Бц исп'] += np.random.randint(0, 19, full_df.shape[0])
full_df['Целевое значение NTMSC на клетку'] = np.random.randint(1600, 2100, full_df.shape[0])
# full_df['Вероятность_Беременности'] = np.random.choice([0.1, 0.3, 0.6, 0.8, 0.95], full_df.shape[0],
#                                                        p=[1./10, 1./10, 1./5, 2./5, 1./5])
full_df['Вероятность_Беременности'] = 0
full_df['Прогноз_Беременности'] = np.random.choice([0, 1], full_df.shape[0], p=[1./4, 3./4])
full_df['Вероятность_Беременности'] = round(abs(full_df['Прогноз_Беременности'] -
                                                np.random.choice([0.1, 0.3, 0.26, 0.18, 0.15],
                                                full_df.shape[0], p=[1./10, 1./10, 1./5, 2./5, 1./5])),2)



full_df = full_df.sample(1000).copy()

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
empty = [x.iloc[0] for x in [df.loc[row].isna().sum(1) for row in df.index]]
minimum_empty = min(empty)
maximum_empty = max(empty)
everything = AllNums()

external_stylesheets = [ {"href": "https://fonts.googleapis.com/css2?"
                        "family=Lato:wght@400;700&display=swap",
                        "rel": "stylesheet"},]



#Запуск самого Дашборда
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

app.config['suppress_callback_exceptions'] = True  #Поработать с этим позже


def check_null(value):
    if str(value) == 'nan':
        return 'Пустая ячейка'
    else:
        return ''


def check_null_outlier(key, value): #TODOs совместить с предыдущей функцией
    if str(value) != 'nan':
        if value > key[0] + key[1] or value < key[0] - key[1]:
            return 'Отклоняется от нормы --'
        return 'Нормальное значение'
    else:
        return 'Пустая ячейка'

#Элемент с таблицей. Вынесен в отдельную функцию, чтобы не дублировать код
def get_table(df):
    return dash_table.DataTable(
            id='datatable-interactivity',
            columns=[ {'name': i, 'id': i, 'deletable': True, 'selectable': True, 'format': Format(nully='N/A')} for i in df.columns],
            data=df.to_dict('records'), style_cell={'whiteSpace': 'normal', 'height': 'auto'}, editable=True,
            filter_action='native', #sort_action='native',
            sort_mode='multi', #column_selectable='multi',
            #row_selectable=False, row_deletable=True,
            selected_columns=[], selected_rows=[], page_action='native',
            page_current=0, page_size=30,
            style_data_conditional = [
            {'if': { 'filter_query': '{} > {} || {} < {}'.format("{"+col+"}", key[0]+key[1], "{"+col+"}", key[0]-key[1]), 'column_id': col },
                'color': 'white','fontWeight': 'bold', 'backgroundColor': 'tomato'} for col, key in col_means_dict.items()]+
            [{'if': {'filter_query': '{{{}}} is nil'.format(col), 'column_id': col},
                'color': 'white', 'fontWeight': 'bold', 'backgroundColor': 'yellow'} for col in df.columns]+
            [{'if': { 'filter_query': '{Прогноз_Беременности} > 0', 'column_id': 'Прогноз_Беременности' },
                        'color': 'white','fontWeight': 'bold', 'backgroundColor': 'green'}]+
            [{'if': { 'filter_query': '{Прогноз_Беременности} < 1', 'column_id': 'Прогноз_Беременности' },
                        'color': 'white','fontWeight': 'bold', 'backgroundColor': 'tomato'}] +
            [{'if': { 'filter_query': '{Вероятность_Беременности} < 0.5', 'column_id': 'Вероятность_Беременности' },
                        'color': 'white','fontWeight': 'bold', 'backgroundColor': 'tomato'}] ,

        tooltip_data=[ #TODOs поправить логику
                         {
                             column: {'value': check_null_outlier(col_means_dict[column], value), 'type': 'markdown'}
                             for column, value in row.items() if column in col_means_dict.keys()
                         } for row in df.to_dict('records')],
            )



server = app.server
app.layout = html.Div([
        html.Div(
        children=[
        html.Div( children=[
        html.Img(src=app.get_asset_url('/images/my_image_clinic.png')), # Подгружаем картинку
        html.H1(children="Pregnancy Analytics", className="header-title", style={"fontSize": "48px", 'color':'black'},),
        html.P(children="Контролируй все процессы сети клиник в одном месте", className="header-description"),]),

        #Подгрузка Данных
        dcc.Upload(id='upload-data', children=html.Div([ 'Перетащите файл сюда или ', html.A('выберите файл на компьютере') ]),
            multiple=False, # Allow multiple files to be uploaded
            style={ 'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'} ),
        html.Div(id='output-data-upload'),],),

        #Кнопки фильтров для таблицы
        html.Div(children=[
            html.Div(children=[ html.Div(children="Регион", className="menu-title"),
                dcc.Dropdown(id="head-region-filter", value="Москва", multi=True, clearable=False, className="dropdown",
                             options=[{"label": 'Все', "value": 'all'}, {"label": 'Москва', "value": 'Москва'},{"label": 'Пермь', "value": 'Пермь'}],
                              ), ]),
            html.Div(children=[ html.Div(children="Группа", className="menu-title"),
                dcc.Dropdown(id="head-Anomal-filter", value="all", clearable=False,searchable=False, className="dropdown",
                             options=[{"label": 'Все', "value": 'all'},
                            {"label": 'Только отклонения', "value": 'anomal'},{"label": 'Только нормальные', "value": 'normal'}],),
                html.Div(children=[html.Div(children="Количество ячеек со значимым отклонением (красные ячейки) в одной строке", className="menu-title"),
                                   dcc.RangeSlider(className="menu-RangeSlider",id="head-AnomalCount-filter",
                                    marks={i: '{}'.format(i) for i in range(0, 26, 3)},
                                                   count=1, min=0, max=25, step=1, value=[0, 18])], ),
                html.Div(children=[html.Div(children="Степень отклонения от нормы (Вы можете подобрать насколько процентов значение в ячейке может отклоняться от среднего значения в данном столбце)", className="menu-title"),
                                    dcc.Slider(className="menu-RangeSlider",id="head-AnomalLevel-filter",#tooltip={"value":'Количество сигм'},
                                    marks={i: '{}'.format(i) for i in range(0, 6, 1)}, min=0, max=5, step=1, value=1)], ),  
                html.Div(children=[
                html.Div(children='Количество пропусков', className='menu-title'),
                dcc.RadioItems(options=[
                    {'label': 'Без пропусков', 'value': 0},
                    {'label': 'Максимум', 'value': maximum_empty},
                    {'label': 'Минимум', 'value': minimum_empty},
                    {'label': 'Не использовать критерий', 'value': -1}
                ], value=0, labelStyle={'display': 'inline-block'}, id='head-EmptyCount-filter')
                ])                                     
                ]),
            html.Div(children=[ html.Div(children="Период времени", className="menu-title"),
                dcc.DatePickerRange(id="head-date-range", min_date_allowed=data["Дата пункции"].min(),
                                    max_date_allowed=data["Дата пункции"].max(),
                                    start_date=data["Дата пункции"].min(),
                                    end_date=data["Дата пункции"].max(), ), ]),
        ], className="head-menu", ),  # head-menu - стили для головной панели над таблицей


        # Кнопка для снятия фильтрации
        html.Button('Снять все фильтры', id='clear-filters', n_clicks=0),

        # Инструкция использования
        html.Div(children=[
            html.Div(children=[
                html.H2(id='use-hint-top', children='Чтобы задать фильтр напишите в пустой строке над необходимым столбцом выбранное значение или диапазон  (взято для примера)', style={'textAlign': 'center'}),
                html.H2(children='=18, либо', style={'textAlign': 'center'},),
                html.H2(children='>18, либо', style={'textAlign': 'center'}),
                html.H2(children='<18, либо', style={'textAlign': 'center'}),
                html.H2(children='18-24 (диапазон)', style={'textAlign': 'center'}),
                html.H2(id='use-hint-bot', children='Если в столбце нечисловые значения отфильтруйте по возможным значениям, пример: СПЛИТ / ПГТ-А', style={'textAlign': 'center'}),
            ]),
            html.Div(children=[
                html.Img(src='/assets/images/tooltip.png', style={'width': '100%', 'height': '100%'})
            ])
        ], style={'display': 'grid', 'grid-template-columns': '80% 20%'}),

        # Таблица
        html.Div(id='table-filtered', children=[
            get_table(df), ]),

        # Фильтры для графика
        html.Div(children=[
            html.Div(children=[ html.Div(children="Город", className="menu-title"),
                dcc.Dropdown(id="region-filter", options=[{"label": 'Moscow', "value": 'Moscow'}],
                             value="Moscow", clearable=False, className="dropdown", ), ]),
            html.Div(children=[ html.Div(children="Type", className="menu-title"),
                dcc.Dropdown(id="doctor-filter", options=[{"label": 'Все', "value": 'Все'}],
                             value="Все", clearable=False, searchable=False, className="dropdown", ), ]),

            html.Div(children=[ html.Div(children="Период времени", className="menu-title"),
                dcc.DatePickerRange(id="date-range", min_date_allowed=data["Дата пункции"].min(),
                                    max_date_allowed=data["Дата пункции"].max(),
                                    start_date=data["Дата пункции"].min(),
                                    end_date=data["Дата пункции"].max(), ), ]), ], className="menu", ),
       # График с инфой из фильтров
        html.Div(children=[
                html.Div(children=dcc.Graph(id="price-chart", config={"displayModeBar": False}),className="card",),
                html.Div(children=dcc.Graph(id="volume-chart", config={"displayModeBar": False}),className="card",),
                html.Div(id='datatable-interactivity-container', className="card"),
        ], className="wrapper",), ])

# Снятие всех фильтров
@app.callback(
    Output('head-AnomalCount-filter', 'value'),
    Output('head-AnomalLevel-filter', 'value'),
    Output('head-EmptyCount-filter', 'value'),
    [Input('clear-filters', 'n_clicks')])
def clear_all_filters(n_clicks):
    return [0, 12], 2, -1 # Здесь оставляем те значения, который являются стандартными


#Подгрузка локальной таблицы
@app.callback(
    Output('datatable-interactivity-container', 'children'),
    [Input('datatable-interactivity', 'derived_virtual_data'),
     Input('datatable-interactivity', 'derived_virtual_selected_rows')])
def update_graphs(rows, derived_virtual_selected_rows):
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    dff = df if rows is None else pd.DataFrame(rows)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    return [ dcc.Graph( id=column,
            figure={ "data": [ {"x": dff["Возраст"], "y": dff[column],
                    "type": "bar", "marker": {"color": colors},} ],  "layout": {
                    "xaxis": {"automargin": True},
                    "yaxis": { "automargin": True, "title": {"text": column}},
                    "height": 250, "margin": {"t": 10, "l": 10, "r": 10}, }, }, )
        for column in ["Эмбриолог ТВП", "Фаза стимуляции", "Получено всего ооцитов"] if column in dff ]


#Функция подгрузки нового файла
def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    print(content_type, content_string)
    try:
        if 'csv' in filename: df = pd.read_csv( io.StringIO(decoded.decode('windows-1251'))) # Assume that the user uploaded a CSV file
        elif 'xls' in filename: df = pd.read_excel(io.BytesIO(decoded)) # Assume that the user uploaded an excel file
    except Exception as e:
        print('Ошибка при подгрузке файлов')
        print(e)
        df = 'None'

        return html.Div([ 'There was an error processing this file.'])
    df.dropna(axis=0, inplace=True, how='all')
    assert len(df) > 0,'Пустой массив'
    df.dropna(axis=1, inplace=True, how='all')
    selected_column_list = []
    filtered_data =df
    level = 1
    print(df.head())
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        get_table(df), #Табличка
        html.Hr(),  # horizontal line
        html.Div('Raw Content'), # For debugging, display the raw contents provided by the web browser
        html.Pre(contents[0:200] + '...', style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all' })
    ])

#Вызов подгузки файла
@app.callback(Output('data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
              State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    print('Подгружаю файл')
    if list_of_contents is not None:
        children = [ parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        return children



# # Фитрация таблицы по цвету аномалий
def update_charts_color(filtered_data, selected_column_list):
    selected_column_list_mask = []
    for col in selected_column_list:
        if col in num_cols:
            anomal_count = np.zeros(filtered_data.shape[0])
            threshold = col_means_dict[col]
            colum_line = filtered_data[col]
            less_line_mask = colum_line < threshold[0] - threshold[1]
            mole_line_mask = colum_line > threshold[0] + threshold[1]
            anomal_count += (less_line_mask | mole_line_mask)

            selected_column_list_mask.append('Количество_отклонений_'+col)
            filtered_data['Количество_отклонений_'+col] = anomal_count

    # filtered_data['Количество_отклонений'] = anomal_count
    if len(selected_column_list_mask)>0:
        filtered_data.sort_values(by=selected_column_list_mask, inplace=True, axis=0, ascending=False)
    return filtered_data

# Фитрация таблицы по количеству аномалий с панели
@app.callback(
    Output("table-filtered", 'children'),
    [Input("datatable-interactivity", "selected_columns"),
    Input("head-region-filter", "value"),
     Input("head-Anomal-filter", "value"),
     Input("head-AnomalLevel-filter", "value"),
     Input("head-AnomalCount-filter", "value"),
     Input("head-EmptyCount-filter", "value"),
     Input("head-date-range", "start_date"),
     Input("head-date-range", "end_date"),], )
def update_charts(selected_column_list, region, anomal, level, anomal_range_list,
                  empty_range_list, start_date, end_date):

    #print(selected_column_list, 'selected_column_list')
    mask = (
            (df["Дата пункции"] >= start_date)
            & (df["Дата пункции"] <= end_date))
    filtered_data = df.loc[mask, :]

    anomal_count = np.zeros(filtered_data.shape[0])

    for col in full_df.columns:
        if col in num_cols:
            threshold = col_means_dict[col]
            colum_line = filtered_data[col]
            less_line_mask = colum_line < threshold[0] - level*threshold[1]
            mole_line_mask = colum_line > threshold[0] + level*threshold[1]
            anomal_count += less_line_mask
            anomal_count += mole_line_mask

    filtered_data['Количество_отклонений'] = anomal_count
    if empty_range_list != -1:
        filtered_data = filtered_data[(filtered_data['Количество_отклонений'] >= anomal_range_list[0])&
                                    (filtered_data['Количество_отклонений'] <= anomal_range_list[1]) &
                                    (filtered_data.isna().sum(1) == empty_range_list)]
    else:
        filtered_data = filtered_data[(filtered_data['Количество_отклонений'] >= anomal_range_list[0])&
                                    (filtered_data['Количество_отклонений'] <= anomal_range_list[1])]


    if anomal == 'normal':
        filtered_data = filtered_data[filtered_data['Количество_отклонений'] == 0]
    elif anomal == 'anomal':
        filtered_data = filtered_data[filtered_data['Количество_отклонений'] > 0 ]

    filtered_data = update_charts_color(filtered_data, selected_column_list)

    return get_table(filtered_data)


#График для аналитики
@app.callback([Output("price-chart", "figure"),
     Output("volume-chart", "figure")],
    [Input("region-filter", "value"),
        Input("doctor-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),],)
def update_charts(region, doctor, start_date, end_date):
    global df
    mask = (
        #(data.region == region) &
        #& (data.type == avocado_type)
         (df["Дата пункции"] >= start_date)
        & (df["Дата пункции"] <= end_date))
    filtered_data = df.loc[mask, :]

    filtered = filtered_data.groupby("Дата пункции", as_index=False).agg(
        average_pregnancy=pd.NamedAgg(column='Прогноз_Беременности', aggfunc='mean'),
        average_embrion=pd.NamedAgg(column='Количество эмбрионов', aggfunc='mean'))

    price_chart_figure = { "data": [ {"x": filtered["Дата пункции"], "y": filtered["average_pregnancy"]*100,
                           "type": "lines", "hovertemplate": "%{y:.2f}<extra></extra>%",},],
                        "layout": {"title": {"text": "Средний процент беременности", "x": 0.05,"xanchor": "left",},
            "xaxis": {"fixedrange": True}, "yaxis": {"ticksuffix": "%", "fixedrange": True}, "colorway": ["#17B897"],
        },}

    volume_chart_figure = {
        "data": [{"x": filtered["Дата пункции"], "y": filtered["average_embrion"],
                "type": "lines",},],
        "layout": {"title": {"text": "Среднее количество эмбрионов","x": 0.05,"xanchor": "left"},
            "xaxis": {"fixedrange": True},
            "yaxis": {"fixedrange": True},
            "colorway": ["#E12D39"],
        },
    }
    return price_chart_figure, volume_chart_figure



if __name__ == '__main__':
    app.run_server(debug=True, port=8050)