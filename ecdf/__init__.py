"""
ecdf
====

Visualize dataframes with echarts.

"""

__all__ = [
    'Echart'
]

import math
import json
import uuid

import numpy as np
import pandas as pd


APPEND_ELEMENT = '''
    $('#{eid}').attr('id','{eid}'+'_old');
    element.append('<div id="{eid}" style="width: {width}px;height:{height}px;page-break-inside: avoid;"></div>');
    '''

FUNC_ELEMENT = '''require(['echarts'],
    function(echarts){
        var myChart = echarts.init(document.getElementById("%s"),"");
        var option = %s;
        myChart.setOption(option);
    });
    '''

DIV_ELEMENT = '<div id="{eid}" class="ecdf" style="width:{width};height:{height};page-break-inside:avoid;"></div>'

OPTION_ELEMENT = '''<script type="text/javascript">
    var myChart = echarts.init(document.getElementById("{eid}"),"");
    var option = {option};
    myChart.setOption(option);
</script>
'''


def update_dict_(base, updates):
    if updates is None:
        return base
    for k, v in updates.items():
        if k in base and isinstance(v, dict):
            update_dict_(base[k], v)
        else:
            base[k] = v
    return base

# def edge(idx: int, nrows: int, ncols: int):
#     ic = idx % ncols
#     ir = idx // ncols
#     edges = {
#         'L': ic == 0,
#         'R': ic == ncols-1,
#         'T': ir == 0,
#         'B': ir ==  nrows-1
#     }
#     edges['H'] = edges['L'] or edges['R']
#     edges['V'] = edges['T'] or edges['B']
#     return edges

def left(idx: int, ncols: int, space: float=0.01):
    return (idx % ncols)/ncols + space

def top(idx: int, nrows: int, ncols: int, space: float=0.1):
    return idx // ncols / nrows + space

def pct(val):
    return '{:.2%}'.format(val)


def nb():
    """
    Inject Javascript to notebook, default using local js.
    This function must be last executed in a cell to produce the Javascript in the output cell
    """
    from IPython.display import Javascript
    js = Javascript("""
    require.config({
        paths: {
            echarts: "https://cdn.bootcss.com/echarts/4.1.0/echarts.min"
        }
    });
    """)
    return js


class Echart:
    def __init__(self, data, xcol=None, xtype=None, title='', subtitle='',
                 width=800, height=500, left='10%', bottom=60, zoomer=False, xlim=None, ylim=None,
                 ytype='value', orient='v', double_precision=3):
        data = data.copy()
        if data.index.name is None:
            data.index.name = 'index'
        x = data.index.name if xcol is None else xcol
        if isinstance(data, pd.Series):
            name = 'value' if data.name is None else data.name
            data = data.to_frame(name)
        data.columns = list(map(str, data.columns))
        ys = list(data.columns)
        data = data.reset_index()
        if xtype is None:
            dtype = data[x].dtype.name
            if dtype == 'datetime64[ns]':
                xtype = 'time'
            elif dtype.startswith('float'):
                xtype = 'value'
            else:
                xtype = 'category'
        self.xtype = xtype
        self.datasets = [data]
        self.x = x
        self.ys = ys
        self.width = width
        self.height = height
        self.orient = orient
        self.double_precision = double_precision

        option = {
            'dataset': ['{{dataset}}'],
            'series': [],
            'title': {'text': title, 'subtext': subtitle,},
            'legend': {'left': 'right'},
            'dataZoom': [],
            'animation': False,
            'grid': [{'left': left, 'bottom': bottom}],
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                    'type': 'cross',
                    'animation': False,
                    'snap': False,
                    'label': {
                        'backgroundColor': '#505765',
                    },
                },
            },
        }
        xa = {'type': xtype}
        ya = {'type': ytype}
        if self.orient == 'h':
            xa, ya =ya, xa
        if xlim is not None:
            a, b = xlim
            xa.update({'min': a, 'max': b})
        if ylim is not None:
            a, b = ylim
            ya.update({'min': a, 'max': b})
        option['xAxis'] = [xa]
        option['yAxis'] = [ya]
        self._option = option
        if zoomer:
            self.add_zoomer()

    @property
    def xAxis(self):
        return self._option['xAxis']
    
    @property
    def yAxis(self):
        return self._option['yAxis']

    def add_zoomer(self, axis='x', idx=0, filter='filter'):
        axis = '{}AxisIndex'.format(axis)
        option = [
            {
                'type': 'slider',
                axis: idx,
                'show': True,
                'realtime': True,
                'start': 0,
                'end': 100,
                'filterMode': filter,
            },
            {
                'type': 'inside',
                axis: idx,
                'show': True,
                'realtime': True,
                'start': 0,
                'end': 100,
                'filterMode': filter,
            },
        ]
        self._option['dataZoom'] += option

    def update_component(self, key: str, index: int=None, **kwargs):
        if key in self._option:
            if index is None:
                if isinstance(self._option[key], dict):
                    update_dict_(self._option[key], kwargs)
                else:
                    for _cfg in self._option[key]:
                        update_dict_(_cfg, kwargs)
            else:
                old = self._option[key][index]
                self._option[key][index] = update_dict_(old, kwargs)
        else:
            append = index is not None
            self.add_component(key, append=append, **kwargs)
        return self

    def add_component(self, key: str, append=False, **kwargs):
        if append:
            if key in self._option:
                if not isinstance(self._option[key], list):
                    self._option[key] = [self._option[key]]
                self._option[key].append(kwargs)
            else:
                self._option[key] = [kwargs]
        else:
            self._option[key] = kwargs
        return self

    def remove_component(self, key: str):
        if key in self._option:
            del self._option[key]
        return self

    def remove_components(self, keys: list):
        for key in keys:
            self.remove_component(key)
        return self

    def update_xAxis(self, index=None, **kwargs):
        self.update_component('xAxis', index=index, **kwargs)
        return self

    def update_yAxis(self, index=None, **kwargs):
        self.update_component('yAxis', index=index, **kwargs)
        return self


    def add_yAxis(self, **kwargs):
        if len(self.yAxis) == 0:
            options = {'type': 'value'}
        else:
            options = {'type': 'value', 'position':'right', 'splitLine': {'show': False}}
        update_dict_(options, kwargs)
        self.add_component('yAxis', append=True, **options)
        return len(self.yAxis) - 1


    def add_series(self, kind: str, y=None, x: str=None, datasetIndex=None, **kwargs):
        if y is None:
            exists = [s['name'] for s in self._option['series']]
            y = [y for y in self.ys if y not in exists]
        elif isinstance(y, str):
            y = [y]
        if datasetIndex is None:
            datasetIndex = list(range(len(self.datasets)))
        elif not isinstance(datasetIndex, (tuple, list)):
            datasetIndex = [datasetIndex]
        for _y in y:
            series = {
                'name': kwargs.get('name', _y),
                'type': kind,
            }
            if kind == 'pie':
                encode = {
                    'itemName': x,
                    'value': _y,
                }
            else:
                encode = {
                    'x': x if x is not None else self.x,
                    'y': _y,
                }
                if self.orient == 'h':
                    encode['x'], encode['y'] = encode['y'], encode['x']
            series['encode'] = encode
            
            for di in datasetIndex:
                series = series.copy()
                series['datasetIndex'] = di
                series['xAxisIndex'] = di
                series['yAxisIndex'] = di
                update_dict_(series, kwargs)
                self._option['series'].append(series)
        return self

    def plot(self, kind, *args, **kwargs):
        funcs = {
            'line': self.line,
            'bar': self.bar,
            'area': self.area,
        }
        return funcs[kind](*args, **kwargs)

    def line(self, y=None, x=None, reference=False, **kwargs):
        if reference:
            kwargs = update_dict_({'lineStyle': {'type': 'dashed', 'width': 1}, 'z':1}, kwargs)
        else:
            _cfg = {
                'symbol': 'none', 
                'lineStyle': {'width': 2,},
                'smooth': True,
                'sampling': 'average',
                }
            kwargs = update_dict_(_cfg, kwargs)
        self.add_series('line', y, x, **kwargs)
        return self

    def bar(self, y=None, x=None, **kwargs):
        if self.orient == 'h':
            fun = self.update_yAxis
        else:
            fun = self.update_xAxis
        fun(splitArea={'show': True, 'interval':0})
        # self.update_component(key='tooltip', axisPointer={'type': 'shadow'})
        self.add_series('bar', y, x, **kwargs)
        return self

    def area(self, y=None, x=None, **kwargs):
        styles = {
            'lineStyle': {'width': 0}, 'areaStyle': {'opacity': 0.8}, 'symbolSize': 0,
        }
        update_dict_(styles, kwargs)
        self.add_series('line', y, x=x, **styles)
        return self

    def by(self, col, ncols=3, link=None, spaces=(0.05, 0.03, 0.25)):
        self.ys = [c for c in self.ys if c != col]
        df = self.data
        by = df.pop(col)
        ncols = min(ncols, len(set(by)))
        if link is not None:
            if link == 'h':
                link = 'x' if link =='y' else 'y'
            self._option['axisPointer'] = {'link': {f'{link}AxisIndex': 'all'}}

        datasets = []
        nrows = math.ceil(len(set(by))/ncols)
        width = 0.95/ncols
        height = 0.9/nrows
        xo = self.xAxis[0]
        yo = self.yAxis[0]

        idx = 0
        self.remove_components(['grid', 'xAxis', 'yAxis', 'title'])
        for g, _df in df.groupby(by):
            datasets.append(_df)
            _grid = {
                'top': pct(top(idx, nrows, ncols, space=spaces[0])),
                'left': pct(left(idx, ncols, space=0.0)),
                'width': pct(width),
                'height': pct(height),
                'show': True,
                'borderWidth': 0,
                'backgroundColor': '#fff',
                'shadowColor': 'rgba(0,0,0,0.3)',
                'shadowBlur': 2,
                'containLabel': True,
            }

            _title = {
                'text': f'{col}={g}',
                'top': pct(top(idx, nrows, ncols, space=spaces[1])),
                'left': pct(left(idx, ncols) + width*spaces[2]),
                'textStyle': {
                    # 'fontSize': 12,
                    'align': 'left',
                    'fontWeight': 'normal',
                }
            }
            _x = xo.copy()
            _x['gridIndex'] = idx
            _y = yo.copy()
            _y['gridIndex'] = idx

            self.add_component('grid', append=True, **_grid)
            self.add_component('xAxis', append=True, **_x)
            self.add_component('yAxis', append=True, **_y)
            self.add_component('title', append=True, **_title)
            idx += 1
        self.datasets = datasets
        return self


    def grid(self, kind='line', columns=None, ncols: int=5, align: str='y', 
             refval=None, series_kwargs=None, grid_kwargs=None, 
             xAxis_kwargs=None, yAxis_kwargs=None, title_kwargs=None):
        if align is None:
            align = ''
        if refval is not None:
            refcol = '__refval__'
            self.data[refcol] = refval
        
        if columns is None:
            columns = [col for col in self.data.columns if col not in (self.x, refcol)]
        minv = np.nanmin(self.data[columns].values)
        maxv = np.nanmax(self.data[columns].values)
        nrows = math.ceil(len(columns)/ncols)
        space = 0.01
        width = 1/ncols - space
        height = 1/nrows - space
        self.remove_components(['grid', 'xAxis', 'yAxis', 'title'])
        for idx, col in enumerate(columns):
            _grid = {
                'top': pct(top(idx, nrows, ncols, space=space)),
                'left': pct(left(idx, ncols, space=space)),
                'width': pct(width),
                'height': pct(height),
                'show': True,
                'borderWidth': 0,
                'backgroundColor': '#fff',
                'shadowColor': 'rgba(0,0,0,0.3)',
                'shadowBlur': 2,
            }
            update_dict_(_grid, grid_kwargs)
            _xAxis = {
                'type': self.xtype,
                'show': False,
                'gridIndex': idx,
            }
            update_dict_(_xAxis, xAxis_kwargs)

            _yAxis = {
                'type': 'value',
                'show': False,
                'gridIndex': idx,
            }
            if 'y' in align:
                _yAxis['min'] = minv
                _yAxis['max'] = maxv
            update_dict_(_yAxis, yAxis_kwargs)
            
            _title = {
                'text': col,
                'textAlign': 'center',
                'top': pct(top(idx, nrows, ncols, 0) + 0.01),
                'left': pct(left(idx, ncols) + width/2),
                'textStyle': {
                    'fontSize': 12,
                    'fontWeight': 'normal',
                }
            }
            update_dict_(_title, title_kwargs)
            self.add_component('grid', append=True, **_grid)
            self.add_component('xAxis', append=True, **_xAxis)
            self.add_component('yAxis', append=True, **_yAxis)
            self.add_component('title', append=True, **_title)
            self.update_component('legend', show=False)
            _series = {
                'xAxisIndex': idx,
                'yAxisIndex': idx,
                'showSymbol': False,
            }
            update_dict_(_series, series_kwargs)
            self.plot(kind, y=col, **_series)
            if refcol is not None:
                _series['reference'] = True
                self.plot(kind, y=refcol, **_series)
        return self

    @property
    def data(self):
        return self.datasets[0]

    @property
    def dataset_json(self):
        jsons = []
        for dataset in self.datasets:
            json = dataset.to_json(double_precision=self.double_precision, orient='split')
            json = json.replace('"columns":', 'dimensions:').replace('"data":', 'source:')
            jsons.append(json)
        return ','.join(jsons)

    @property
    def json(self):
        if len(self.yAxis) == 0:
            self.add_yAxis()
        if len(self._option['series']) == 1:
            self.update_component('legend', show=False)
        if len(self._option['series']) / len(self._option['grid']) >= 5 and 'selected' not in self._option['legend']:
            cols = [s['name'] for s in self._option['series']]
            selected = [cols[0], cols[int(len(cols)/2-1)], cols[-1]]
            self.select(selected)
        jsonstr = json.dumps(self._option, ensure_ascii=False)
        jsonstr = jsonstr.replace('"{{dataset}}"', self.dataset_json)
        return jsonstr

    def select(self, selected=None):
        legend = self._option['legend']
        if 'data' in legend:
            cols = legend['data']
        else:
            cols = set([s['name'] for s in self._option['series']])
        visible = {col: True if col in selected else False for col in cols}
        self.update_component('legend', selected=visible)

    def _repr_javascript_(self):
        eid = str(uuid.uuid4())
        prep = APPEND_ELEMENT.format(
            eid=eid, width=self.width, height=self.height)
        func = FUNC_ELEMENT % (eid, self.json)
        return ''.join([prep, func])

    def js(self, eid=None):
        if eid is None:
            eid = str(uuid.uuid4())
        script = OPTION_ELEMENT.format(eid=eid, option=self.json)
        return script

    def div(self, eid=None, width='100%', height='90%'):
        if eid is None:
            eid = str(uuid.uuid4())
        div = DIV_ELEMENT.format(eid=eid, width=width, height=height)
        script = OPTION_ELEMENT.format(eid=eid, option=self.json)
        return '\n'.join([div, script])


def ecplot(self, kind='line', title='', orient='v', **kwargs):
    ec = Echart(self, title=title, orient=orient)
    return ec.plot(kind=kind, **kwargs)


try:
    import pandas as pd
    if not hasattr(pd.DataFrame, 'ecplot'):
        pd.DataFrame.ecplot = ecplot
        pd.Series.ecplot = ecplot
except Exception:
    pass
