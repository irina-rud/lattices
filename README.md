# Курсовая работа по курсу Теории Решёток. 
## Задание https://github.com/fdrstrok/lazyfca15/blob/master/old_description.pdf

Для задания по Lazy FCA использовалcя датасет SPECT, содержащий данные по позитронной томографии сердца (22 бинарных признака, отвечающие поддиагнозам, целевой признак также бинарный).

## Краткое описание осбенностей датасета:
*	дисбаланс классов при малом числе наблюдений (всего 267 примеров, из них только 55 отрицательные)
* рекомендованные обучающая выборка содерит по 40 примеров положительных и отрицательных данных, а рекомендованная тестовая выборка только 17 отрицательных примеров.
* практически любое «правило», полученное пересечением признаков тестового объекта и объектов и обучающей выборки, находит контрпример

## Описание решения:
Для отнесения объекта к положительному или отрицательному классу использовалась агрегационная функция на основе достоверности  ![equation](http://www.sciweavers.org/upload/Tex2Img_1544007232/render.png)
