# LR3
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
- Загребин Данила Павлович
- РИ212702
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | # | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения.
Ход работы:
·	Создайте новый пустой 3D проект на Unity.
·	Скачайте папку с ML агентом. Вы найдете ее в облаке с исходными файлами к лабораторной работе – ml-agents-release_19.
·	В созданный проект добавьте ML Agent, json – файлы
· Добавление объектов: куб, сфера и пол.
![image](https://user-images.githubusercontent.com/114522298/200927349-445e87af-8eed-4228-acc4-24b757a2454e.png)

·	Если все сделано правильно, то во вкладке с компонентами (Components) внутри Unity вы увидите строку ML Agent.
![image](https://user-images.githubusercontent.com/114522298/200926935-7b96a049-2729-46d7-ac8e-ac987d041e42.png)
·	Далее запускаем Anaconda Prompt для возможности запуска команд через консоль.
·	Далее пишем серию команд для создания и активации нового ML-агента, а также для скачивания необходимых библиотек:
o	mlagents 0.28.0;
o	torch 1.7.1;

·Подключение C# скрипт-файла к сфере:
![image](https://user-images.githubusercontent.com/114522298/200927802-d7d4859c-929e-4b00-9d85-847c56110bca.png)

```
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
private Rigidbody rBody;
// Start is called before the first frame update
void Start()
{
rBody = GetComponent<Rigidbody>();
}
public Transform Target;
public override void OnEpisodeBegin()
{
    if (this.transform.localPosition.y < 0)
    {
    this.rBody.angularVelocity = Vector3.zero;
    this.rBody.velocity = Vector3.zero;
    this.transform.localPosition = new Vector3(0, 0.5f, 0);
    }


Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
}
public override void CollectObservations(VectorSensor sensor)
{
sensor.AddObservation(Target.localPosition);
sensor.AddObservation(this.transform.localPosition);
sensor.AddObservation(rBody.velocity.x);
sensor.AddObservation(rBody.velocity.z);
}
public float forceMultipler = 10;
public override void OnActionReceived(ActionBuffers actionsBuffer)
{
Vector3 controlSignal = Vector3.zero;
controlSignal.x = actionsBuffer.ContinuousActions[0];
controlSignal.z = actionsBuffer.ContinuousActions[1];
rBody.AddForce(controlSignal * forceMultipler);

float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

if (distanceToTarget < 1.42f)
{
SetReward(1.0f);
EndEpisode();
}
else if (this.transform.localPosition.y < 0)
{
EndEpisode();
}
}
}

```

· Запуск 8 симуляций сцен для обучения
![2022-11-09 23-04-58](https://user-images.githubusercontent.com/114522298/200928146-b7129e91-c7a6-4edd-af2f-820e045c5ac3.gif)

· Запуск сцены после обучения
![1 2](https://user-images.githubusercontent.com/114522298/200928294-d92edd07-8208-4aaa-9c18-f50d637a746c.gif)
 

## Задание 2
### Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта по ссылке. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных сфере.

```
behaviors:
  RollerAgent: //имя агента
    trainer_type: ppo //тип тренера
    hyperparameters:
      batch_size: 10 //количество попыток в каждой итерации
      buffer_size: 100 //количество попыток до изменения схемы движения
      learning_rate: 3.0e-4 // изменчивость рейтинга обучения
      beta: 5.0e-4 // степень случайности выбора схемы движения
      epsilon: 0.2 //стабильность обновления обучения
      lambd: 0.99 //
      num_epoch: 3 // количество опытов необходимых для обновления схемы движения
      learning_rate_schedule: linear //параметр определяет функцию роста рейтинга обучения
    network_settings: //сетевые настройки
      normalize: false //параметр для нормалицации входных данных
      hidden_units: 128 // количество скрытых единиц неронной сети
      num_layers: 2 // определяет количество скрытых слоев нейронной сети
    reward_signals:// параметр влияющий на систему вознаграждения в процессе обучения
      extrinsic:// коэффициент для внешних вознаграждений
        gamma: 0.99 // коэффициент для будущих вознаграждений
        strength: 1.0 // коэффициент влияющий на степень вознаграждения
    max_steps: 500000 //максимальное количество экспериментов 
    time_horizon: 64 // коэффициент влияющий на то, сколько должен длиться эксперимент пережде че заносить его в буфер опыта
    summary_freq: 10000 //количество опытов перед опубликованием текущей статистики
```
Decision Requester - компонент запрашивает с помощью которого агент запрашивает решение каждый раз когда вызывается метод Decision Requester(). Добавив этот компонент к объекту можно запрашивать решения через определённый интервал времени.
Behavior Parameters - скрипт по которому определяется способ обучения. С помощью него можно задать условия при которых будет начинаться новая итерация или заканчиваться предыдущая. 

## Выводы

В лабораторной работе я познакомился на практике с машинным обучением. Научился создавать создавать и активировать MlAgent, работать с библиотекми mlagents 0.28.0, torch 1.7.1. Также я научился создавать сцены в Unity а также размножать их с помощью префабов для ускорения обучения агента. 
Сиситемы машинного обучения могут сделать игру более интересной, подгоняя уровень сложности под пользователя сохраняя баланс между сложностью и простотой. Также такие системы могут оптимизировать процесс организации жизнедеятельности npc, систематизировать трафик на дорогах. 

| Plugin | README |
| ------ | ------ |
| Dropbox | [plugins/dropbox/README.md][PlDb] |
| GitHub | [plugins/github/README.md][PlGh] |
| Google Drive | [plugins/googledrive/README.md][PlGd] |
| OneDrive | [plugins/onedrive/README.md][PlOd] |
| Medium | [plugins/medium/README.md][PlMe] |
| Google Analytics | [plugins/googleanalytics/README.md][PlGa] |

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
