import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, ParameterGrid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def plot_grid_res(params, res, res_train=[], res_time=[], scoring='', name=''):
    '''
    Функция строит графики зависимости метрики исследуемой модели от значений гиперпараметров.
    Если изменяется только один гиперпараметр, строит график-линию (функция plot_line),
    если два - график heatmap (функция plot_heatmap)
    
    Вход:
    - params - словарь изменяемых гиперпараметров
    - res - метрика по результатам кросс-валидации моделей 
    - res_train - метрика на обучающей выборки (если отсутствует, соответствующий график не выводится)
    - res_time - время обучения моделей (если отсутствует, соответствующий график не выводится)
    - scoring - название метрики для подписей оси y
    - name - имя моделей для подписей графиков
    '''
    if not len(params):
        return

    params_values = list(params.values())
    params_keys = list(params.keys())

    # линия для одного параметра
    if len(params_keys)==1:
        if len(res_train):
            plot_line(params_values[0], res, res_train, title=name, xlabel=params_keys[0], ylabel=scoring)
        else:
            plot_line(params_values[0], res, title=name, xlabel=params_keys[0], ylabel=scoring)
        if len(res_time):
            plot_line(params_values[0], res_time, title=name, xlabel=params_keys[0], ylabel='fit time')

    # тепловая карта для двух параметров
    elif len(params_keys)==2:
        # располагаем параметры в алфавитном порядке, чтобы совпало с выходом GridSearchCV
        if params_keys[1] < params_keys[0]:
            params_values = [params_values[1], params_values[0]]
            params_keys = [params_keys[1], params_keys[0]]
        len_p1 = len(params_values[0])
        len_p2 = len(params_values[1])
        res = np.reshape(res, (len_p1, len_p2)).T
        plot_heatmap(params_values[0], params_values[1], res, title=name+' '+scoring, label=params_keys)
    #elif len(params_keys)>2:
    #    print('can\'t plot results')


def plot_line(x, y, y2=[], title='', xlabel='', ylabel=''):
    '''
    Функция строит графики зависимости выбранных метрик исследуемой модели от значения одного гиперпараметра
    Вход:
    - х - значения аргумента (гиперпараметра)
    - y - значения функции 1 (метрика при кросс-валидации)
    - y2 - значения функции 2 (метрика на обучающей выборке)
    - title - подпись графика (название модели)
    - xlabel - подпись оси Х (название изменяемого гиперпараметра)
    - ylabel - подпись оси Y (название метрики)
    '''
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(x, y, marker='.', label='valid')
    if len(y2):
        ax.plot(x, y2, marker='.', label='train')
    ax.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def plot_heatmap(x1, x2, y, title='', label=['','']):
    '''
    функция строик график типа "heatmap" зависимости качества исследуемой модели от значений двух гиперпараметров
    х1 - значения гиперпараметра 1
    х2 - значения гиперпараметра 2
    y - значение метрики в виде одномерного массива (результат работы функции GridSearchCV.cv_results_['...'])
    title - название модели (подпись графика)
    label - массив с названиями двух гиперпараметров (подписи осей Х и Y)
    '''
    
    xsize = len(x1) if len(x1)<16 else 16
    ysize = len(x2) if len(x2)<16 else 16
    fig, ax = plt.subplots(figsize=(xsize,ysize))
    im = ax.imshow(y, cmap='coolwarm')
    plt.title(title)
    
    cax = fig.add_axes([ax.get_position().x1+0.05, ax.get_position().y0, 0.02, ax.get_position().height])
    plt.colorbar(im, cax=cax)

    ax.set_xticks(range(len(x1)))
    ax.set_yticks(range(len(x2)))
    ax.set_xticklabels(x1)
    ax.set_yticklabels(x2)
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])

    y = y.round(3)
    for i in range(len(x1)):
        for j in range(len(x2)):
            text = ax.text(i, j, y[j, i], ha="center", va="center", color="k", fontsize=8) 
    plt.show()



def sklearn_gridCV(model, X, y, scoring, params={}, name='', fit_train=False, plot_time=False, plot=True, cv=5):
    '''
    Кросс-валидация на базе GridSearchCV
    
    Вход:
    - model - исследуемая модель, 
    - X - признаки (features), 
    - y - целевой признак (target), 
    - scoring - метрика, 
    - params - словарь с набором гиперпараметров, 
    - name - имя модели (только для подписи графиков)
    - fit_train - выводить ли результат на обучающей выборке
    - plot_time - выводить ли график времени обучения, 
    - plot - выводить ли графики с результатом, 
    - cv - кол-во фолдов
    
    Выход: сводная таблица с результатами для каждой точки сетки
    '''
    
    grid = GridSearchCV(model, params, cv=cv, scoring=scoring, return_train_score=fit_train, n_jobs=-1, verbose=10)
    grid.fit(X, y)
    
    # разбираемся со знаком метрик
    metric_error = False
    if all(x < 0 for x in grid.cv_results_['mean_test_score']):
        metric_error = True
        if isinstance(scoring, str):
            scoring = scoring[4:]
        grid.cv_results_['mean_test_score'] = -grid.cv_results_['mean_test_score']
        if fit_train:
            grid.cv_results_['mean_train_score'] = -grid.cv_results_['mean_train_score']
        grid.best_score_ = -grid.best_score_

    # заполняем массивы результатами
    res_cv = grid.cv_results_['mean_test_score']
    res_cv_std = grid.cv_results_['std_test_score']
    res_train = grid.cv_results_['mean_train_score'] if fit_train else []
    res_train_std = grid.cv_results_['std_train_score'] if fit_train else []
    res_time = grid.cv_results_['mean_fit_time'] if plot_time else []

    # графики
    if plot:
        plot_grid_res(params, scoring=scoring, res=res_cv, res_train=res_train, res_time=res_time)

    # отображение лучшего результата
    valid_mean = grid.best_score_
    valid_std = grid.cv_results_['std_test_score'][grid.best_index_]
    print('Best valid score: %.3f (%.3f)' % (valid_mean, valid_std))
    if fit_train:
        train_mean = grid.cv_results_['mean_train_score'][grid.best_index_]
        train_std = grid.cv_results_['std_train_score'][grid.best_index_]
        print('Train score:      %.3f (%.3f)' % (train_mean, train_std))
    if len(params):
        print('Best params:', grid.best_params_)

    # возвращаем таблицу с отчетом
    if len(params):
        grid_list = list(ParameterGrid(params))
        res_table = pd.DataFrame(index=grid_list, data={'cv score': res_cv, 'cv std': res_cv_std})
        res_table['time'] = grid.cv_results_['mean_fit_time']
        if fit_train:
            res_table['train score'] = res_train
            res_table['train std'] = res_train_std  
        #if metric_error:
            #return res_table.sort_values('cv score').round(3)
        #return res_table.sort_values('cv score', ascending=False).round(3)
        return res_table.round(3)


def get_feature_importance(model, X, y, fit=False):
    '''
    Вывод значимости признаков для деревянных (model.feature_importances_)
    или линейных моделей (model.coef_)
    '''
    if fit:
        model.fit(X, y)
        
    try:
        # вариант для pandas DataFrame
        cols = X.columns
    except:
        # вариант для numpy array
        cols = range(X.shape[1]) 
        
    feature_importance = pd.DataFrame(index=cols)
    try:
        # вариант для деревьев
        feature_importance['importance'] = model.feature_importances_
    except:
        # вариант для линейных моделей
        feature_importance['lr coef'] = model.coef_
        feature_importance['abs coef'] = abs(feature_importance['lr coef'])
        return feature_importance.sort_values('abs coef', ascending=False).round(3)
    return feature_importance.sort_values('importance', ascending=False).round(3)


def lgb_cv(lgb_data, scoring, N=100, params={}, cat_cols=[], feature_importance=False):

    '''
    Кросс-валидация для LightGBM с выводом графика изменения метрики от итераций
    
    Вход:
    - lgb_data - датасет LightGBM
    - scoring - целевая мерика 
    - N - количество итераций
    - params - словарь с параметрами модели
    - cat_cols - список категориальных столбцов
    - feature_importance - нужно ли выводить таблицу значимости признаков
    
    Выход: таблица значимости признаков (если feature_importance=True)
    '''

    # только чтобы избежать надоедливого warning
    if not len(cat_cols):
        cat_cols = 'auto'

    # stratified для регрессии должен быть False (по умолчанию True)
    stratified = False
    if 'objective' in params.keys():
        if params['objective'] in ['binary', 'multiclass']:
            stratified = True
    
    # early stopping round
    esr = None if N<=100 else int(N/10)
    
    # кросс-валидация из LightGBM
    res = lgb.cv(params, lgb_data, nfold=5, metrics=metric, num_boost_round=N, 
                 categorical_feature=cat_cols, stratified=stratified,
                 verbose_eval=int(N/10), early_stopping_rounds=esr, return_cvbooster=feature_importance, seed=123)
    
    # отрисовка графика
    res_name = metric+'-mean'
    plt.plot(res[res_name])
    plt.xlabel('iteration')
    plt.ylabel(metric)
    plt.grid()
    
    # отображение лучшего результата
    if metric in ['auc', 'accuracy', 'f1']:
        print('Best res:', max(res[res_name]).round(5))
        print('Best iteration:', np.argmax(res[res_name]))
    elif metric in ['rmse']:
        print('Best res:', min(res[res_name]).round(5))
        print('Best iteration:', np.argmin(res[res_name]))
    else:
        print('unknown metric, specify is it score or error in <lgb_lib.py: cv>')
    print('Last res:', res[res_name][-1].round(5))
    
    # вывод таблицы значимости признаков
    if feature_importance:
        return cvb_feature_importance(res['cvbooster'])


def lgb_gridCV(data, scoring=[], model_params={}, grid_params={}, N=100, cv=5, cat_cols=[], 
             fit_train=False, verbose=None, plot_time=False):
    '''
    Поиск гиперпараметров по сетке для LightGBM (аналог GridSearchCV)
    
    Вход:
    - lgb_data - датасет LightGBM
    - scoring - целевая мерика
    - model_params - словарь с фиксированными гиперпараметрами модели
    - grid_params - словарь с перебираемыми гиперпараметрами
    - N - количество итераций
    - cv - количество фолдов
    - cat_cols - список категориальных столбцов
    - verbose - количество промежуточных результатов, выводимых при кросс-валидации каждого набора гиперпараметров 
    - plot_time - отображать ли затраченное на обучение время
    - feature_importance - нужно ли выводить таблицу значимости признаков
    
    Вывод: сводная таблица с резлуьтатами для каждого набора гиперпараметров
    
    !!! Не реализовано на данный момент !!!
    - вывод результата на обучающей выборке (входной параметр fit_train)
    '''

    # только чтобы избежать надоедливого warning
    if not cat_cols:
        cat_cols = 'auto'

    metric_error = True if scoring in ['rmse'] else False
    
    # stratified для регрессии должен быть False (по умолчанию True)
    stratified = False
    if 'objective' in model_params.keys():
        if model_params['objective'] in ['binary', 'multiclass']:
            stratified = True
    
    # пересчитываем частоту выдачи сообщений
    if verbose:
        verbose = int(N/verbose)

    # заготовки массивов для записи 
    grid_res = []
    grid_iter = []
    grid_time = []

    # early stopping round
    esr = None if N<=100 else int(N/10)

    # перебор гиперпараметров по сетке
    grid = list(ParameterGrid(grid_params))
    for g in grid:
        model_params.update(g)
        
        time1 = time.time()
        res = lgb.cv(model_params, data, nfold=cv, metrics=scoring, num_boost_round=N, 
                     categorical_feature=cat_cols, verbose_eval=verbose, stratified=stratified,
                     early_stopping_rounds=esr, seed=123)
        grid_time.append(time.time() - time1)

        res_name = scoring+'-mean'
        best_res = min(res[res_name]).round(5) if metric_error else max(res[res_name]).round(5)
        best_iter = np.argmin(res[res_name]) if metric_error else np.argmax(res[res_name])
        print('params=%s  res=%f  iter=%d' % (g, best_res, best_iter))

        grid_res.append(best_res)
        grid_iter.append(best_iter)

    # вывод графиков
    if plot_time:
        plot_grid_res(grid_params, grid_res, scoring=scoring, res_time=grid_time)
    else:
        plot_grid_res(grid_params, grid_res, scoring=scoring)

    # отображение лучшего результата
    final_idx = np.argmin(grid_res) if metric_error else np.argmax(grid_res)
    print('Best cv score:', grid_res[final_idx])
    if len(grid_params) > 0:
        print('Best params:', grid[final_idx])
    print('Best iteration:', grid_iter[final_idx])
    
    # вывод итоговой таблицы
    res_table = pd.DataFrame(index=grid, data={'cv score': grid_res, 'iter': grid_iter, 'time': grid_time})
    if fit_train:
        res_table['train'] = res_train
    return res_table.round(3)


def cvb_feature_importance(cvb, top=0):
    '''
    Вывод таблицы значимости признаков при кросс-валидации на LightGBM
    
    Вход:
    - cvb - cvBooster
    - top - количество выводимых признаков их числе наиболее значимых (0 - все)
    '''
    importance = np.zeros(len(cvb.boosters[0].feature_name()), dtype='int')
    
    for model in cvb.boosters:
        importance += model.feature_importance()
        
    feature_imp = pd.DataFrame(data={'feature': cvb.boosters[0].feature_name(), 'importance': importance})
    feature_imp.sort_values(by='importance', ascending=False, inplace=True)
    feature_imp.reset_index(drop=True, inplace=True)
    
    if top:
        return feature_imp.head(top)
    else:
        return feature_imp



def compare_features(feature_imp, data, field):
    cols = [i for i in feature_imp['feature'].values if field in i]
    
    ncol = 2
    nrow = len(cols)
    plt.figure(figsize=(20,nrow*5))
    bins=20
    
    for i in range(len(cols)):
        ax = plt.subplot(nrow, ncol, i+1)
        sns.histplot(data=data, x=cols[i], bins=bins, stat='density', hue='flag', common_norm=False)
        pos = feature_imp[feature_imp['feature']==cols[i]].index.values
        imp = feature_imp[feature_imp['feature']==cols[i]]['importance']
        ax.set_xlabel('#%d - %s - %d' % (pos,cols[i],imp))
    plt.show()