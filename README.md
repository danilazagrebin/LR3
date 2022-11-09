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

·

## Задание 2
### Реализовать запись в Google-таблицу набора данных, полученных с помощью линейной регрессии из лабораторной работы № 1. 
![image](https://user-images.githubusercontent.com/114522298/195174429-63e5768e-6ae2-48a0-a3c6-73f9652ec571.png)
![image](https://user-images.githubusercontent.com/114522298/195174459-35bd70f5-149e-498e-8258-06496ffbc93a.png)
#Import the required modules, numpy for calculation, and Matplotlib for drawing

Взял код из первой лабораторной и преобразовал его, чтобы связать с google таблицей.
```
import numpy as np
import gspread
gc = gspread.service_account(filename='lr-2-2-60c96be9dd05.json')
sh = gc.open("LR-2-2")
# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)


# define data, and change list to array
x = [3,21,22,34,54,34,55,67,89,99]
x = np.array(x)
y = [2,22,24,65,79,82,55,130,150,199]
y = np.array(y)

#The basic linear regression model is wx+ b, and since this is a two-dimensional space, the model is ax+ b

def model(a, b, x):
    return a*x + b

#Tahe most commonly used loss function of linear regression model is the loss function of mean variance difference
def loss_function(a, b, x, y):
    num = len(x)
    prediction=model(a,b,x)
    return (0.5/num) * (np.square(prediction-y)).sum()

#The optimization function mainly USES partial derivatives to update two parameters a and b
def optimize(a,b,x,y):
    num = len(x)
    prediction = model(a,b,x)
    #Update the values of A and B by finding the partial derivatives of the loss function on a and b
    da = (1.0/num) * ((prediction -y)*x).sum()
    db = (1.0/num) * ((prediction -y).sum())
    a = a - Lr*da
    b = b - Lr*db
    return a, b

#iterated function, return a and b
def iterate(a,b,x,y,times):
    for i in range(times):
        a,b = optimize(a,b,x,y)
    return a,b

#Initialize parameters and display
a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a,b = iterate(a,b,x,y,100)
prediction=model(a,b,x)
loss = loss_function(a, b, x, y)
print(a,b,loss)

mon = list(range(1,20))
i = 0
while i <= len(mon):
    i += 1
    if i == 0:
        continue
    else:
        a, b = iterate(a, b, x, y, 100)
        prediction = model(a, b, x)
        loss = loss_function(a, b, x, y)
        tempInf = str(loss)
        tempInf = tempInf.replace('.', ',')
        sh.sheet1.update(('A' + str(i)), str(i))
        sh.sheet1.update(('B' + str(i)), str(tempInf))
        print(tempInf)
```

## Выводы

В лабораторной работе я познакомился с такими сервисами, как Google Cloud, Google sheets. Также я научился связывать программу написанную на языке Python с таблицей google Sheets, еще я научился реализовывать совместную работу и передачу данных в связке Python - Google-Sheets – Unity.

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
